import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
from skimage.transform import resize
from sklearn.metrics import recall_score, precision_score
import os
import time
import argparse
import pandas as pd
import numpy as np
import random
from models import *
from sam import *

parser = argparse.ArgumentParser()

# Dataset Arguments
parser.add_argument('--data_dir', default='../data/kvasir-dataset-v2-cropped-224', type=str)
parser.add_argument('--resolution', default=224, type=int)
parser.add_argument('--train_test_split', default=0.8, type=float)

# Training Arguments
parser.add_argument('--arch', default='resnet20', type=str)
parser.add_argument('--layers', default=1, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr', default=0.05, type=float)
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--optim', default="SGD", type=str, choices=['SGD', 'SAM'])

# Logging Arguments
parser.add_argument('--log_dir', default='../logs', type=str)
parser.add_argument('--save_weights', default=1, type=int)


class KvasirDataset(Dataset):
    def __init__(self, dir, resolution, is_train, train_test_split = 0.8):
        self.dir = dir
        self.data = None #data.float().to(device)
        self.labels = None #labels.float().to(device)
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
        self.resolution = resolution
        self.is_train = is_train
        self.train_test_split = train_test_split
        if self.is_train==True:
            self.transform = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                                #transforms.RandomVerticalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        self._load_images()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) == list:
            data = torch.cat([self.transform(self.data[i]) for i in idx], dim=0)
        else:
            data = self.transform(self.data[idx])
        return (data, self.labels[idx])

    def _load_images(self):
        print("--> Loading Dataset")
        tic = time.time()
        images = []
        labels = []
        for cl_idx, cl in enumerate(self.classes):
            parent_path = args.data_dir + "\\" + cl
            image_paths = [parent_path+"\\"+name for name in os.listdir(parent_path)]
            if self.is_train == True:
                image_paths = image_paths[:int(len(image_paths)*self.train_test_split)]
            else:
                image_paths = image_paths[int(len(image_paths)*self.train_test_split):]

            for img_count, path in enumerate(image_paths):
                #resized_image = resize(io.imread(path).astype(np.float16)/255, (self.resolution, self.resolution), anti_aliasing=True)
                resized_image = ((io.imread(path)/255).reshape((self.resolution, self.resolution, 3)).astype(np.float16))
                images.append(resized_image)
                labels.append(cl_idx)
            print("Loaded Class (", cl_idx, "): ", cl)
        self.data = images
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

def recall(output, target):
    _, preds = torch.max(output, 1)
    preds = preds.detach().cpu().tolist()
    target = target.detach().cpu().tolist()
    return recall_score(preds, target, average='micro')

def precision(output, target):
    _, preds = torch.max(output, 1)
    preds = preds.detach().cpu().tolist()
    target = target.detach().cpu().tolist()
    return precision_score(preds, target, average='micro')

def get_run_name(args):
    random_string = ''.join(random.choice('0123456789abcdefghijklmnopqrstuvwxyz') for _ in range(6))
    run_name = 'arch=' + str(args.arch) + "_lr=" + str(args.lr) + "_epochs=" + str(args.epochs) + \
                '_res=' + str(args.resolution) + "_split=" + str(args.train_test_split) + \
                '_seed=' + str(args.seed) + "_layers=" + str(args.layers) + "_optim=" + args.optim + \
                '_' + random_string
    return run_name

def get_confusion_matrix(num_classes, dataloader, model):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloader):
            inputs, classes = inputs.cuda().float(), classes.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix.cpu().detach().numpy()

def main(args):
    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Create Datasets/Dataloaders
    train_dataset = KvasirDataset(args.data_dir, args.resolution, is_train=True, train_test_split=args.train_test_split)
    test_dataset = KvasirDataset(args.data_dir, args.resolution, is_train=False, train_test_split=args.train_test_split)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    # Create Neural Network Objects
    model = models_dict[args.arch](num_layers=args.layers).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'SAM':
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Data Logging Variables
    train_acc, train_loss, train_time = [], [], []
    train_f1, train_recall, train_precision = [], [], []
    test_acc, test_loss, test_time = [], [], []
    test_f1, test_recall, test_precision = [], [], []
    final_confusion_matrix = None #
    run_name = get_run_name(args)

    for epoch in range(args.epochs):

        # Training
        train_tic = time.time()
        sub_train_acc = AverageMeter()
        sub_train_prec = AverageMeter()
        sub_train_rec = AverageMeter()
        sub_train_loss = AverageMeter()

        model.train()
        for input, target in train_loader:
            input, target = input.cuda().float(), target.cuda()
            optimizer.zero_grad()
            output = model(input).cuda()
            if args.optim == 'SGD':
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()
            elif args.optim == 'SAM':
                # first forward-backward step
                enable_running_stats(model)
                output = model(input).cuda()
                loss = criterion(output, target)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward step
                disable_running_stats(model)
                criterion(model(input).cuda(), target).mean().backward()
                optimizer.second_step(zero_grad=True)

            sub_train_acc.update(accuracy(output.float().data, target)[0].item(), input.size(0))
            sub_train_rec.update(recall(output.float().data, target), input.size(0))
            sub_train_prec.update(precision(output.float().data, target), input.size(0))
            sub_train_loss.update(loss.float().item(), input.size(0))

            del input, target
            
        scheduler.step()
        train_acc.append(sub_train_acc.avg)
        train_recall.append(sub_train_rec.avg)
        train_precision.append(sub_train_prec.avg)
        train_f1.append(2 * (train_precision[epoch] * train_recall[epoch]) / (train_precision[epoch] + train_recall[epoch]))
        train_loss.append(sub_train_loss.avg)
        train_time.append(time.time() - train_tic)

        # Testing
        test_tic = time.time()
        sub_test_acc = AverageMeter()
        sub_test_rec = AverageMeter()
        sub_test_prec = AverageMeter()
        sub_test_loss = AverageMeter()

        model.eval()
        with torch.no_grad():
            for input, target in test_loader:
                input, target = input.cuda().float(), target.cuda()
                output = model(input).cuda()
                loss = criterion(output, target)

                sub_test_acc.update(accuracy(output.float().data, target)[0].item(), input.size(0))
                sub_test_rec.update(recall(output.float().data, target), input.size(0))
                sub_test_prec.update(precision(output.float().data, target), input.size(0))
                sub_test_loss.update(loss.float().item(), input.size(0))
                
                del input, target

        test_acc.append(sub_test_acc.avg)
        test_recall.append(sub_test_rec.avg)
        test_precision.append(sub_test_prec.avg)
        test_f1.append(2 * (test_precision[epoch] * test_recall[epoch]) / (test_precision[epoch] + test_recall[epoch]))
        test_loss.append(sub_test_loss.avg)
        test_time.append(time.time() - test_tic)

        print('EPOCH: {0}\n'
            'Train | Acc: {1:.3f}\tLoss: {2:.3f}\tTime: {3:.3f}\n'
            'Test  | Acc: {4:.3f}\tLoss: {5:.3f}\tTime: {6:.3f}'.format(epoch+1,
            train_acc[epoch], train_loss[epoch], train_time[epoch],
            test_acc[epoch], test_loss[epoch], test_time[epoch]))

    # Confusion Matrix
    final_confusion_matrix = get_confusion_matrix(len(test_dataset.classes), test_loader, model)

    # Create log directory
    if not os.path.exists(args.log_dir + "\\" + run_name):
        os.makedirs(args.log_dir + "\\" + run_name)

    # ==> Save data to xlsx
    writer = pd.ExcelWriter(args.log_dir + "\\" + run_name + "\\results.xlsx", engine='xlsxwriter')

    # Train/Test Metric
    df1 = pd.DataFrame(data = {
        "train_acc": train_acc, 
        "train_loss": train_loss,
        "train_time": train_time,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "test_time": test_time,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1}, index=list(range(1, args.epochs+1)))
    df1.to_excel(writer, sheet_name='Train and Test Metrics', index = True)

    # Final Confusion Matrix
    df2 = pd.DataFrame(data = {k: final_confusion_matrix[:, i].tolist() for i, k in enumerate(test_dataset.classes)}, index=test_dataset.classes)
    df2.to_excel(writer, sheet_name='Confusion Matrix', index = True)
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