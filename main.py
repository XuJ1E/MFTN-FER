# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2023/11/2 20:55
# Author     ：XuJ1E
# version    ：python 3.8
# File       : main.py
"""
import os
import torch
import argparse
import datetime
from torch.backends import cudnn
from torchvision import transforms, datasets
from torchsampler import ImbalancedDatasetSampler
from util import LabelSmoothingCrossEntropy, RecorderMeter, RecorderMeter1, SAM, train_one_epoch, evaluate, save_checkpoint
from models import Model
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")
parser = argparse.ArgumentParser()
parser.add_argument('--data_type', default='RAFDB', choices=['RAFDB', 'AffectNet_7', 'ExpW'], type=str, help='dataset optional')
parser.add_argument('--dataset', default='./data/RAFDB/', type=str, help='dataset for training')
parser.add_argument('--log', type=str, default='./log/', help='log file path')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model.pth')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model_best.pth')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int, metavar='N')
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')

parser.add_argument('--lam', default=0.45, type=float, help='hyper_param of loss adjust.')
parser.add_argument('--lr', default=2.5e-4, type=float, metavar='LR', dest='lr')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('--print-freq', default=20, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('--evaluate', default=None, type=str, help='evaluate model on test set')
parser.add_argument('--gpu', type=str, default='0,1', help='number of GPUs which per node.')
args = parser.parse_args()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    best_acc = 0
    print('Training time: ' + now.strftime("%m-%d %H:%M"))

    model = Model(num_classes=7, pretrained=True, drop_path_rate=0.25)
    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1).cuda()

    if args.optimizer == 'adamw':
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")

    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, adaptive=False, )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    recorder = RecorderMeter(args.epochs)
    recorder1 = RecorderMeter1(args.epochs)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            recorder1 = checkpoint['recorder1']
            best_acc = best_acc.to()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True
    if args.evaluate is None:
        if args.data_type == 'RAFDB':
            train_dataset = datasets.ImageFolder(os.path.join(args.dataset, 'train'),
                                                 transforms.Compose([transforms.Resize((224, 224)),
                                                                     transforms.RandomHorizontalFlip(),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                          std=[0.229, 0.224, 0.225]),
                                                                     transforms.RandomErasing(scale=(0.02, 0.1))]))
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=args.workers,
                                                       pin_memory=True,
                                                       drop_last=True)

        else:
            train_dataset = datasets.ImageFolder(os.path.join(args.dataset, 'train'),
                                                 transforms.Compose([transforms.Resize((224, 224)),
                                                                     transforms.RandomHorizontalFlip(),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                          std=[0.229, 0.224, 0.225]),
                                                                     transforms.RandomErasing(p=1, scale=(0.05, 0.07))]))
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       sampler=ImbalancedDatasetSampler(train_dataset),
                                                       batch_size=args.batch_size,
                                                       shuffle=False,
                                                       num_workers=args.workers,
                                                       pin_memory=True,
                                                       drop_last=True)

    val_dataset = datasets.ImageFolder(os.path.join(args.data, 'val'),
                                       transforms.Compose([transforms.Resize((224, 224)),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])]))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.evaluate is not None:
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}'".format(args.evaluate))
            checkpoint = torch.load(args.evaluate)
            best_acc = checkpoint['best_acc']
            best_acc = best_acc.to()
            print(f'best_acc:{best_acc}')
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.evaluate))
        evaluate(model=model, data_loader=val_loader, args=args)
        return

    for epoch in range(args.start_epoch, args.epochs):

        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print('Now lr: ', current_learning_rate)
        txt_name = args.log + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Now lr: ' + str(current_learning_rate) + '\n')

        # train for one epoch
        train_acc, train_los = train_one_epoch(model, train_loader, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_acc, val_los, output, labels, D = evaluate(model, val_loader, args)

        scheduler.step()

        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder1.update(output, labels)

        curve_name = time_str + 'cnn.png'
        recorder.plot_curve(os.path.join(args.log, args.data_type, curve_name))

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        print('Current best accuracy: ', best_acc.item())

        # if is_best:
        #     matrix = D

        # print('Current best matrix: ', matrix)

        txt_name = os.path.join(args.log, args.data_type) + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder1': recorder1,
                         'recorder': recorder}, is_best, args)


if __name__ == '__main__':
    main()
