import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
from skimage.transform import resize
import os
import time
import argparse
import pandas as pd
import numpy as np
import random
from models import *

parser = argparse.ArgumentParser()

# Dataset Arguments
parser.add_argument('--data_dir', default='../data/kvasir-dataset-v2-cropped', type=str)
parser.add_argument('--resolution', default=112, type=int)
parser.add_argument('--train_test_split', default=0.8, type=float)

# Training Arguments
parser.add_argument('--arch', default='resnet20', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--seed', default=1000, type=int)

# Logging Arguments
parser.add_argument('--log_dir', default='../logs', type=str)
parser.add_argument('--save_weights', default=1, type=int)


class KvasirDataset(Dataset):
    def __init__(self, dir, resolution, is_train, train_test_split = 0.8):
        self.dir = dir
        self.data = None #data.float().to(device)
        self.labels = None #labels.float().to(device)
        self.classes = None
        self.classes_to_omit = [0, 1, 6]
        self.resolution = resolution
        self.is_train = is_train
        self.train_test_split = train_test_split
        self.transform = transforms.Compose([transforms.Resize((self.resolution, self.resolution)),
                                             #transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self._load_images()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.data[idx], self.labels[idx])

    def _load_images(self):
        print("--> Loading Dataset")
        tic = time.time()
        self.classes = [
            #'dyed-lifted-polyps',
            #'dyed-resection-margins',
            'esophagitis',
            'normal-cecum',
            'normal-pylorus',
            'normal-z-line',
            'polyps',
            'ulcerative-colitis'
        ]

        images = []
        labels = []
        for cl_idx, cl in enumerate(self.classes):
            parent_path = args.data_dir + "\\" + cl
            image_paths = [parent_path+"\\"+name for name in os.listdir(parent_path)]
            if self.is_train == True:
                image_paths = image_paths[:int(len(image_paths)*self.train_test_split)]
            else:
                image_paths = image_paths[int(len(image_paths)*self.train_test_split):]

            for path in image_paths:
                #resized_image = resize(io.imread(path), (self.resolution, self.resolution), anti_aliasing=True)
                resized_image = (io.imread(path).astype(np.float64) / 255).reshape((3, self.resolution, self.resolution))
                #fliped_hor_resized = np.flip(resized_image, axis=1)
                #fliped_ver_resized = np.flip(resized_image, axis=2)
                images.append(resized_image)
                #images.append(fliped_hor_resized)
                #images.append(fliped_ver_resized)
                labels.append(cl_idx)
                #labels.append(cl_idx)
                #labels.append(cl_idx)
            print("Loaded Class (", cl_idx, "): ", cl)
        self.data = torch.Tensor(images)
        self.labels = torch.Tensor(labels).long()
        print("--> Dataset Load Time: ", time.time() - tic, "s")


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().cuda()
    correct = pred.eq(target.cuda().view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_run_name(args):
    random_string = ''.join(random.choice('0123456789abcdefghijklmnopqrstuvwxyz') for _ in range(6))
    run_name = 'arch=' + str(args.arch) + "_lr=" + str(args.lr) + "_epochs=" + str(args.epochs) + \
                '_res=' + str(args.resolution) + "_split=" + str(args.train_test_split) + \
                '_seed=' + str(args.seed) + "_" + random_string
    return run_name

def main(args):
    # Reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Create Datasets/Dataloaders
    train_dataset = KvasirDataset(args.data_dir, args.resolution, is_train=True, train_test_split=args.train_test_split)
    test_dataset = KvasirDataset(args.data_dir, args.resolution, is_train=False, train_test_split=args.train_test_split)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    # Create Neural Network Objects
    model = models_dict[args.arch]().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Data Logging Variables
    train_acc, train_loss, train_time = [], [], []
    test_acc, test_loss, test_time = [], [], []
    run_name = get_run_name(args)

    for epoch in range(args.epochs):

        # Training
        train_tic = time.time()
        sub_train_acc = AverageMeter()
        sub_train_loss = AverageMeter()

        model.train()
        for input, target in train_loader:
            input, target = input.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(input).cuda()
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            sub_train_acc.update(accuracy(output.float().data, target)[0].item(), input.size(0))
            sub_train_loss.update(loss.float().item(), input.size(0))

        scheduler.step()
        train_acc.append(sub_train_acc.avg)
        train_loss.append(sub_train_loss.avg)
        train_time.append(time.time() - train_tic)

        # Testing
        test_tic = time.time()
        sub_test_acc = AverageMeter()
        sub_test_loss = AverageMeter()

        model.eval()
        with torch.no_grad():
            for input, target in test_loader:
                input, target = input.cuda(), target.cuda()
                output = model(input).cuda()
                loss = criterion(output, target)

                sub_test_acc.update(accuracy(output.float().data, target)[0].item(), input.size(0))
                sub_test_loss.update(loss.float().item(), input.size(0))

        test_acc.append(sub_test_acc.avg)
        test_loss.append(sub_test_loss.avg)
        test_time.append(time.time() - test_tic)

        print('EPOCH: {0}\n'
            'Train | Acc: {1:.3f}\tLoss: {2:.3f}\tTime: {3:.3f}\n'
            'Test  | Acc: {4:.3f}\tLoss: {5:.3f}\tTime: {6:.3f}'.format(epoch+1,
            train_acc[epoch], train_loss[epoch], train_time[epoch],
            test_acc[epoch], test_loss[epoch], test_time[epoch]))

    # Create log directory
    if not os.path.exists(args.log_dir + "\\" + run_name):
        os.makedirs(args.log_dir + "\\" + run_name)

    # Save data to xlsx
    df = pd.DataFrame(data = {
        "train_acc": train_acc, 
        "train_loss": train_loss,
        "train_time": train_time,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "train_time": test_time})
    writer = pd.ExcelWriter(args.log_dir + "\\" + run_name + "\\results.xlsx", engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Test Acc and Loss', index = False)
    writer.save() 

    # Save model weights
    if args.save_weights == 1:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, args.log_dir + "\\" + run_name + "\\model.tar")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)