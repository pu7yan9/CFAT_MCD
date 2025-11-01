import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x

class PreNorm(nn.Module):
    def  __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x


class SpatialFrequencyCoRepresentation(nn.Module):
    def __init__(self, size=3, input_size=13, scale_factor=2):
        super().__init__()

        self.kernel_size = size
        self.input_size = input_size
        self.scale_factor = scale_factor

        pad_total = size - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        self.pad = (pad_left, pad_right, pad_left, pad_right)

        self.unfold = nn.Unfold(kernel_size=(size, size), stride=(1, 1))

        dim_patch = size * size
        linear_in = dim_patch * 2
        linear_out = scale_factor ** 2
        self.linear = nn.Linear(linear_in, linear_out)

        fold_kernel = (scale_factor, scale_factor)
        fold_stride = (scale_factor, scale_factor)
        out_spatial = input_size * scale_factor
        self.fold = nn.Fold(
            kernel_size=fold_kernel,
            stride=fold_stride,
            output_size=(out_spatial, out_spatial)
        )

        self.pixel_unshuffle = nn.PixelUnshuffle(scale_factor)

        x = torch.linspace(0., size - 1, steps=size)
        u = x.unsqueeze(1)
        C = torch.sqrt(torch.threshold(-u + 1, 0.5, 2) / size).repeat(1, size)
        G = torch.cos(u * (2 * x + 1) * np.pi / (2 * size)) * C
        self.register_buffer('G_kernel', G)

    def forward(self, x):
        N, C, H, W = x.shape
        x_pad = F.pad(x, self.pad, mode='replicate')
        patches = self.unfold(x_pad)
        L = H * W
        patches = patches.transpose(1, 2).contiguous().view(N, L, C, self.kernel_size, self.kernel_size)
        spa = patches.view(N, L, C, self.kernel_size * self.kernel_size)
        fre = self.dct(patches).view(N, L, C, self.kernel_size * self.kernel_size)

        cat = torch.cat([spa, fre], dim=-1)
        out = self.linear(cat)
        out = out.view(N, L, C * (self.scale_factor ** 2))

        out = self.fold(out.transpose(1, 2).contiguous())
        out = self.pixel_unshuffle(out)
        return out

    def dct(self, x):
        return torch.matmul(torch.matmul(self.G_kernel, x), self.G_kernel.T)


class MTT(nn.Module):
    def __init__(self, in_channels, atrous_rates, depth, patch_size, kernel_size):
        super(MTT, self).__init__()
        dim = 256
        head = 4
        convs = []
        sfcms = []
        in_convs = []
        en_trans = []
        de_trans = []
        pos_ens = []

        self.token_len = 4
        for i in range(depth):
            convs.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels//4, 3, padding=atrous_rates[i], dilation=atrous_rates[i], bias=False),
                nn.BatchNorm2d(in_channels//4),
                nn.ReLU())
            )
            sfcms.append(SpatialFrequencyCoRepresentation(size=kernel_size, input_size=patch_size))
            in_convs.append(nn.Conv2d(dim, self.token_len, kernel_size=(1, 1), bias=False))
            en_trans.append(Transformer(dim=dim, depth=1, heads=head, dim_head=dim, mlp_dim=dim, dropout=0))
            de_trans.append(TransformerDecoder(dim=dim, depth=1, heads=head, dim_head=dim, mlp_dim=dim, dropout=0, softmax=True))
            pos_ens.append(nn.Parameter(torch.randn(1, self.token_len, dim)).cuda())

        self.convs = nn.ModuleList(convs)
        self.sfcms = nn.ModuleList(sfcms)
        self.in_convs = nn.ModuleList(in_convs)
        self.en_trans = nn.ModuleList(en_trans)
        self.de_trans = nn.ModuleList(de_trans)
        self.pos_ens = pos_ens
        self.project = nn.Sequential(
            nn.Conv2d(depth * in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def _forward_semantic_tokens(self, x, conv_s):
        b, c, h, w = x.shape
        spatial_attention = conv_s(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_transformer(self, x, pos_embedding, transformer):
        x += pos_embedding
        x = transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m, transformer_decoder):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def forward(self, x):
        residual = x
        res = []
        for conv, sfcm, in_conv, en_tran, de_tran, pos_en in zip(self.convs, self.sfcms, self.in_convs, self.en_trans, self.de_trans,
                                                           self.pos_ens):
            x_ = conv(x)
            x_ = sfcm(x_)
            tokens = self._forward_semantic_tokens(x_, in_conv)
            tokens = self._forward_transformer(tokens, pos_en, en_tran)
            x_ = self._forward_transformer_decoder(residual, tokens, de_tran)
            res.append(x_)
        res = torch.cat(res, dim=1)
        return self.project(res)


class MTT_Without_SFCR(nn.Module):
    def __init__(self, in_channels, atrous_rates, depth, patch_size, kernel_size):
        super(MTT_Without_SFCR, self).__init__()
        dim = 256
        head = 4
        convs = []
        in_convs = []
        en_trans = []
        de_trans = []
        pos_ens = []

        self.token_len = 4

        for i in range(depth):
            convs.append(nn.Sequential(
                nn.Conv2d(in_channels, dim, 3, padding=atrous_rates[i], dilation=atrous_rates[i], bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU())
            )
            in_convs.append(nn.Conv2d(dim, self.token_len, kernel_size=(1, 1), bias=False))
            en_trans.append(Transformer(dim=dim, depth=1, heads=head, dim_head=dim, mlp_dim=dim, dropout=0))
            de_trans.append(TransformerDecoder(dim=dim, depth=1, heads=head, dim_head=dim, mlp_dim=dim, dropout=0, softmax=True))
            pos_ens.append(nn.Parameter(torch.randn(1, self.token_len, dim)).cuda())

        self.convs = nn.ModuleList(convs)
        self.in_convs = nn.ModuleList(in_convs)
        self.en_trans = nn.ModuleList(en_trans)
        self.de_trans = nn.ModuleList(de_trans)
        self.pos_ens = pos_ens
        self.project = nn.Sequential(
            nn.Conv2d(depth * in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def _forward_semantic_tokens(self, x, conv_s):
        b, c, h, w = x.shape
        spatial_attention = conv_s(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_transformer(self, x, pos_embedding, transformer):
        x += pos_embedding
        x = transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m, transformer_decoder):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def forward(self, x):
        residual = x
        res = []
        for conv, in_conv, en_tran, de_tran, pos_en in zip(self.convs, self.in_convs, self.en_trans, self.de_trans,
                                                           self.pos_ens):
            x_ = conv(x)
            tokens = self._forward_semantic_tokens(x_, in_conv)
            tokens = self._forward_transformer(tokens, pos_en, en_tran)
            x_ = self._forward_transformer_decoder(residual, tokens, de_tran)
            res.append(x_)
        res = torch.cat(res, dim=1)
        return self.project(res)


class CFAT(nn.Module):
    def __init__(self, n_channels, n_classes, patch_size=15, kernel_size=3, depth=3):
        super(CFAT, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # self.aspp = MTT_Without_SFCR(in_channels=256, atrous_rates=[2, 4, 6], depth=depth, patch_size=patch_size, kernel_size=kernel_size)
        self.aspp = MTT(in_channels=256, atrous_rates=[2, 4, 6], depth=depth, patch_size=patch_size, kernel_size=kernel_size)

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=(7, 7)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.aspp(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x






