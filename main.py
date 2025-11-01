import torch
import torch.nn as nn
import torch.utils.data.dataloader
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from model import CFAT
from dataset import HCD
import argparse

parser = argparse.ArgumentParser(description='CFAT_MCD', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=int, default=2, help='dataset')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
parser.add_argument('--patch_size', type=int, default=15, help='patch_size')
parser.add_argument('--kernel_size', type=int, default=3, help='kernel_size')
parser.add_argument('--num_sample', type=int, default=100, help='window_size')
parser.add_argument('--depth', type=int, default=3, help='depth')
parser.add_argument('--pseudo', type=str, default='IRGMcS', help='pseudo')
parser.add_argument('--network', type=str, default='CFAT', help='network')
parser.add_argument('--epoch', type=int, default=20, help='epoch')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
DATASET = args.dataset
DATASET_NAME = ['Italy', 'Yellow', 'Shuguag', 'Glources2', 'Glources1', 'California', 'France']
print('Dataset:', DATASET_NAME[DATASET-1])
Net = args.network
Epoch = args.epoch
print('Network:', Net)
P_tra = args.pseudo
print('Pseudo label:', P_tra)
Patch_size = args.patch_size
print('Patch_size:', Patch_size)
Kernel_size = args.kernel_size
print('Kernel_size:', Kernel_size)
Num_sample = args.num_sample
print('Num_sample:', Num_sample)
Depth = args.depth
print('Depth:', Depth)

transforms_set = transforms.Compose([transforms.ToTensor()])
transforms_result = transforms.ToPILImage()

train_data = HCD(train=True,
                 dataset_id=DATASET,
                 p_tra = P_tra,
                 transform=transforms_set,
                 patch_size=Patch_size,
                 num = Num_sample)
test_data = HCD(train=False,
                dataset_id=DATASET,
                p_tra = P_tra,
                transform=transforms_set,
                patch_size=Patch_size)

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=256,
                                          shuffle=False)


model = CFAT(n_channels=6, n_classes=1, patch_size=Patch_size, kernel_size=Kernel_size, depth=Depth)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()


def confusion_matrix(true_value, output_data):
    data_size = output_data.shape[0]
    true_value = torch.squeeze(true_value)
    output_data = torch.heaviside(torch.squeeze(output_data), torch.tensor([0], dtype=torch.float32, device='cuda'))
    union = torch.clamp(true_value + output_data, 0, 1)
    intersection = true_value * output_data
    true_positive = int(intersection.sum())
    true_negative = data_size - int(union.sum())
    false_positive = int((output_data - intersection).sum())
    false_negative = int((true_value - intersection).sum())
    return true_positive, true_negative, false_positive, false_negative


def save_visual_result(output_data, img_sequence, is_label=False):
    if is_label:
        output_data = torch.heaviside(torch.squeeze(output_data), torch.tensor([0], dtype=torch.float32, device='cuda'))
    else:
        output_data = torch.squeeze(output_data)
    output_data = output_data.cpu().clone()
    dims = len(output_data.shape)
    if dims > 2:
        batch_size = output_data.shape[0]
        for i in range(batch_size):
            image = transforms_result(output_data[i])
            img_sequence.append(image)
    else:
        image = transforms_result(output_data)
        img_sequence.append(image)
    return img_sequence


def save_attention_visual_result(output_data):
    img_sequence = []
    output_data = torch.squeeze(output_data.cpu().clone())
    batch_size, channel_size = output_data.shape[0], output_data.shape[1]
    for i in range(batch_size):
        for j in range(channel_size):
            image = transforms_result(output_data[i][j])
            img_sequence.append(image)
    return img_sequence


def evaluate(tp, tn, fp, fn):
    oa, recall, precision, f1, false_alarm, missing_alarm, CIOU, UCIOU, MIOU, KC = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    tp, tn, fp, fn = float(tp), float(tn), float(fp), float(fn)
    oa = (tp + tn) / (tp + tn + fp + fn)

    if (tp + fn) > 0:
        recall = tp / (tp + fn)

    if (tp + fp) > 0:
        precision = tp / (tp + fp)

    if (recall + precision) > 0:
        f1 = 2 * ((precision * recall) / (precision + recall))

    if (tn + fp) > 0:
        false_alarm = fp / (tn + fp)

    if (tp + fn) > 0:
        missing_alarm = fn / (tp + fn)
    if (tp + fp + fn) > 0:
        CIOU = tp / (tp + fp + fn)
    if (tn + fp + fn) > 0:
        UCIOU = tn / (tn + fp + fn)
    MIOU = (CIOU + UCIOU) / 2

    total = tp + tn + fp + fn
    Po = (tp + tn) / total if total > 0 else 0
    Pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total ** 2) if total > 0 else 0
    KC = (Po - Pe) / (1 - Pe) if (1 - Pe) != 0 else 0

    return oa, recall, precision, f1, false_alarm, missing_alarm, CIOU, UCIOU, MIOU, KC


def train(train_loader_arg, model_arg, criterion_arg, optimizer_arg, the_epoch):
    model_arg.cuda()
    model_arg.train()
    with tqdm(total=len(train_loader_arg), desc='Train Epoch #{}'.format(the_epoch + 1)) as t:
        for batch_idx, (img_1, img_2, label) in tqdm(enumerate(train_loader_arg)):
            img_1, img_2, label = img_1.cuda(), img_2.cuda(), label.cuda()
            output = model_arg(torch.cat([img_1, img_2], dim=1))
            loss = criterion_arg(output, label)
            optimizer_arg.zero_grad()
            loss.backward()
            optimizer_arg.step()
            t.set_postfix({'lr': '%.5f' % optimizer_arg.param_groups[0]['lr'],
                           'loss': '%.4f' % loss.detach().cpu().data})
            t.update(1)


def test(test_loader_arg, model_arg, the_epoch):
    tp, tn, fp, fn = 0, 0, 0, 0
    oa, recall, precision, f1, false_alarm, missing_alarm, ciou, uciou, miou, kc= 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    model_arg.cuda()
    model_arg.eval()
    with tqdm(total=len(test_loader_arg), desc='Test Epoch #{}'.format(the_epoch + 1)) as t:
        for batch_idx, (img_1, img_2, label) in tqdm(enumerate(test_loader_arg)):
            img_1, img_2, label = img_1.cuda(), img_2.cuda(), label.cuda()
            output = model_arg(torch.cat([img_1, img_2], dim=1))
            tp_tmp, tn_tmp, fp_tmp, fn_tmp = confusion_matrix(label, output)
            tp += tp_tmp
            tn += tn_tmp
            fp += fp_tmp
            fn += fn_tmp
            if batch_idx > 10:
                oa, recall, precision, f1, false_alarm, missing_alarm, ciou, uciou, miou, kc = evaluate(tp, tn, fp, fn)
            t.set_postfix({'acc': oa,
                           'f1': '%.4f' % f1,
                           'KC': '%.4f' % kc,
                           'recall': '%.4f' % recall,
                           'precision': '%.4f' % precision,
                           'false alarm': '%.4f' % false_alarm,
                           'missing alarm': '%.4f' % missing_alarm,
                           'CIOU': '%.4f' % ciou,
                           'UCIOU': '%.4f' % uciou,
                           'MIOU': '%.4f' % miou})
            t.update(1)
    if (the_epoch + 1) >= 1:
        iou_sequence.append(ciou)
        f = open(Net + '_' + str(DATASET_NAME[DATASET-1]) + '_' + P_tra + '_DP' + str(Depth) + '_PS' + str(Patch_size) + '_KS' + str(Kernel_size) +  '.txt', 'a')
        f.write("\"epoch\":\"" + "{}\"\n".format(the_epoch + 1))
        f.write("\"oa\":\"" + "{}\"\n".format(oa))
        f.write("\"KC\":\"" + "{}\"\n".format(kc))
        f.write("\"f1\":\"" + "{}\"\n".format(f1))
        f.write("\"recall\":\"" + "{}\"\n".format(recall))
        f.write("\"precision\":\"" + "{}\"\n".format(precision))
        f.write("\"false alarm\":\"" + "{}\"\n".format(false_alarm))
        f.write("\"missing alarm\":\"" + "{}\"\n".format(missing_alarm))
        f.write("\"CIOU\":\"" + "{}\"\n".format(ciou))
        f.write("\"UCIOU\":\"" + "{}\"\n".format(uciou))
        f.write("\"MIOU\":\"" + "{}\"\n".format(miou))
        f.write("\"max_iou\":\"" + str(max(iou_sequence)) + ' epoch:' + str(
            iou_sequence.index(max(iou_sequence)) + 1) + '\n')
        f.write('\n')
        f.close()
        print('max_iou:' + str(max(iou_sequence)) + ' epoch:' + str(iou_sequence.index(max(iou_sequence)) + 1) + '\n')
        if ciou >= max(iou_sequence):
            if not os.path.isdir('./save'):
                os.makedirs('./save')
            torch.save(model.state_dict(), './save/best_{}_{}_{}_DP{}_PS{}_KS{}_{}.pth'.format(Net, str(DATASET_NAME[DATASET-1]), P_tra, Depth, Patch_size, Kernel_size, '{:.5f}'.format(f1)))


iou_sequence = []
for epoch in range(Epoch):
    train(train_loader_arg=train_loader,
          model_arg=model,
          criterion_arg=criterion,
          optimizer_arg=optimizer,
          the_epoch=epoch)
    if epoch >= 0:
        test(test_loader_arg=test_loader,
             model_arg=model,
             the_epoch=epoch)
