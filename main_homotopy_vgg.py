from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import vgg_homotopy
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import csv


model_names = sorted(name for name in vgg_homotopy.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg_homotopy.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg13',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg13)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    # help='number of total epochs to run')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--auto-find', default=True,
                    help='Find Idle GPU automatically')
parser.add_argument('-l', '--nargs-float-type', nargs='+', type=float)
parser.add_argument('-L', '--penalty', nargs='+', type=float)


global args, best_prec1, val_acc, train_acc, train_loss, val_loss
args = parser.parse_args()
if args.auto_find == True:
    for t in range(8):
        if os.system('nvidia-smi -q -i '+str(t)+' -d performance | grep "Idle                        : Active"') == 0:
            # os.system('export CUDA_VISIBLE_DEVICES='+str(t))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(t)
            break
args.cuda = not args.no_cuda and torch.cuda.is_available()
best_prec1 = 0
val_acc, train_acc, train_loss, val_loss = [], [], [], []
print('L_list: ', args.nargs_float_type)
print('penalty parameters: ', args.penalty)


# Check the save_dir exists or not
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

model = vgg_homotopy.__dict__[args.arch]()

# model.features = torch.nn.DataParallel(model.features)
model.cuda()

# optionally resume from a checkpoint
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.evaluate, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

cudnn.benchmark = True

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

# define loss function (criterion) and pptimizer
criterion = nn.CrossEntropyLoss().cuda()

if args.half:
    model.half()
    criterion.half()

optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

if args.evaluate:
    validate(val_loader, model, criterion)

def train(train_loader, model, criterion, optimizer, epoch, L):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)
        if args.half:
            input_var = input_var.half()

        # compute output
        output_1 = model(input_var,'1')
        output_2 = model(input_var,'2')
        # loss = criterion(output_1, target_var)
        loss = criterion(output_1, target_var) + criterion(output_2, target_var)
        if args.arch == 'vgg11':
            conv_list = [0,3,6,8,11,13,16,18] # vgg11
        elif args.arch == 'vgg13':
            conv_list = [0,2,5,7,10,12,15,17,20,22]  # vgg13
        elif args.arch == 'vgg16':
            conv_list = [0,2,5,7,10,12,14,17,19,21,24,26,28]  # vgg16
        else:
            conv_list = [0,2,5,7,10,12,14,16,19,21,23,25,28,30,32,34]  # vgg19
        for z in conv_list: 
            length = len(model.features[z].weight)//2
            loss += args.penalty[0]*L*torch.dist(model.features_1[z].weight[:length],
                                 # torch.ones(model.features_1[z].weight[:length].shape).cuda()/1000)
                                 torch.zeros(model.features_1[z].weight[:length].shape).cuda())

            loss += args.penalty[0]*L*torch.dist(model.features_1[z].bias[:length],
                                 torch.zeros(model.features_1[z].bias[:length].shape).cuda())

            loss += args.penalty[0]*L*torch.dist(model.features[z].weight[length:],
                                 # torch.ones(model.features[z].weight[length:].shape).cuda()/1000)
                                 torch.zeros(model.features[z].weight[length:].shape).cuda())

            loss += args.penalty[0]*L*torch.dist(model.features[z].bias[length:],
                                 torch.zeros(model.features[z].bias[length:].shape).cuda())


            loss += args.penalty[1]/L*torch.dist(model.features[z].weight,
                                     model.features_1[z].weight)
            loss += args.penalty[1]/L*torch.dist(model.features[z].bias, model.features_1[z].bias)
        for z in [1,4,6]:
            loss += args.penalty[1]/L*torch.dist(model.classifier[z].weight,
                                     model.classifier_1[z].weight)
            loss += args.penalty[1]/L*torch.dist(model.classifier[z].bias,
                                     model.classifier_1[z].bias)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output_1 = output_1.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output_1.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    train_loss.append(losses.avg.detach().cpu().numpy())
    train_acc.append(top1.avg.detach().cpu().numpy())

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)

        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    val_acc.append(top1.avg.detach().cpu().numpy())
    val_loss.append(losses.avg.detach().cpu().numpy())

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    start = time.clock()
    n_epoch = 0
    # L_list = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.01]
    # L_list = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    L_list = args.nargs_float_type
    for L in L_list:
        print('L: ',L)
        if L == L_list[-1]:
            args.epochs = 600
        for epoch in range(1, args.epochs + 1):
            if L == L_list[-1]:
                n_epoch += 1
            adjust_learning_rate(optimizer, n_epoch)
            train(train_loader, model, criterion, optimizer, epoch, L)
            prec1 = validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            # is_best = prec1 > best_prec1
            # best_prec1 = max(prec1, best_prec1)
            # save_checkpoint({
                # 'epoch': epoch + 1,
                # 'state_dict': model.state_dict(),
                # 'best_prec1': best_prec1,
            # }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)

    # Save the results
    result_dir = './hta_result/'+args.arch+'/acc_'+str(val_acc[-1])
    os.mkdir(result_dir)
    csvfile = open(result_dir+'/train_loss.csv' , 'a+', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(train_loss)
    csvfile.close()
    csvfile = open(result_dir+'/train_acc.csv' , 'a+', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(train_acc)
    csvfile.close()
    csvfile = open(result_dir+'/val_loss.csv' , 'a+', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(val_loss)
    csvfile.close()
    csvfile = open(result_dir+'/val_acc.csv' , 'a+', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(val_acc)
    csvfile.close()
