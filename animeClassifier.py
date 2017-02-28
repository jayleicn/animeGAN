from __future__ import print_function
import os
import time
import random
import argparse
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, default=12, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nEpoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.0002')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--evaluate'  , action='store_true', help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--start_epoch', type=int, default=0, help="epoch to start with (to continue training)")
parser.add_argument('--outDir', default='.', help='folder to output images and model checkpoints')

arg_list = [
    '--dataRoot', '/home/jielei/data/danbooru-faces-classification',
    '--workers', '12',
    '--batchSize', '256',
    '--imageSize', '64',
    '--nEpoch', '25',
    '--lr', '0.01',
    '--cuda', 
    '--ngpu', '2',
    '--pretrained', '',
    '--start_epoch', '0',
    #'--evaluate',
    '--outDir', 'results-classification'
]

opt = parser.parse_args(arg_list)
print(opt)

try:
    os.makedirs(opt.outDir)
except OSError:
    pass


cudnn.benchmark = True

if opt.cuda:
    if not torch.cuda.is_available():
        opt.cuda = False
else:
    if torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

opt.manualSeed = random.randint(1,10000) # fix seed, a scalar
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)

# Data loading
trainDir = os.path.join(opt.dataRoot, 'train')
valDir = os.path.join(opt.dataRoot, 'val')
mean = (0.5,0.5,0.5) 
std = (0.5,0.5,0.5)
# common classifier should use the mean and std calculated from the data,
# here we simply move the images to the region [-1,1]

train_loader = torch.utils.data.DataLoader(
    dset.ImageFolder(
    root=trainDir,
    transform=transforms.Compose([
            # transforms.Scale(opt.imageSize),
            # transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std), # bring images to (-1,1)
        ])),
    batch_size=opt.batchSize,
    shuffle=True, num_workers=opt.workers)

val_loader = torch.utils.data.DataLoader(
    dset.ImageFolder(
    root=valDir,
    transform=transforms.Compose([
            # transforms.Scale(opt.imageSize),
            # transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std), # bring images to (-1,1)
        ])),
    batch_size=opt.batchSize,
    shuffle=False, num_workers=opt.workers)

# Model definition
class _animeNet_AlexNet(nn.Module):
    def __init__(self, ngpu, num_class=126):
        super(_animeNet_AlexNet, self).__init__()
        self.ngpu = ngpu
        self.features = nn.Sequential(
            # 96*96
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 48*48
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 24*24
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 12*12
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # 6*6
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(True),
            nn.Linear(4096,4096),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(4096,num_class)
        )
        
    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        x = nn.parallel.data_parallel(self.features, input, gpu_ids)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
net = _animeNet_AlexNet(opt.ngpu)
net.apply(weights_init)

if opt.pretrained != '':
    print('loading pretrained net form %s' % opt.pretrained )
    net.load_state_dict(torch.load(opt.pretrained))
print(net)

criterion = nn.CrossEntropyLoss()

if opt.cuda:
    net.cuda()
    criterion.cuda()

optimizer = optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=1e-5)

def main():
    if opt.evaluate:
        validate(val_loader, net, criterion)
        return
    
    log_path = os.path.join(opt.outDir, 'train.log')
    with open(log_path, 'w') as f:
        f.write('train_loss\ttrain_top1\tval_loss\tval_top1\n')

    best_prec1 = 0
    for epoch in range(opt.start_epoch, opt.nEpoch):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        loss, prec1 = train(train_loader, net, criterion, optimizer, epoch)
        line_to_write = "{:02.4f}\t{:02.4f}\t".format(loss, prec1)

        # evaluate on validation set
        loss, prec1 = validate(val_loader, net, criterion)
        line_to_write =  line_to_write + "{:02.4f}\t{:02.4f}\n".format(loss, prec1)

        with open(log_path, 'a') as f:
            f.write(line_to_write)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': net.__class__.__name__,
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, outDir =opt.outDir)


def train(train_loader, net, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = net(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 5 == 0:
            print('Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg


def validate(val_loader, net, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    net.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = net(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 5 == 0:
            print('Test: [{0}/{1}] '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return losses.avg, top1.avg


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, outDir='.', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(outDir, filename))
    if is_best:
        shutil.copyfile(os.path.join(outDir, filename), os.path.join(outDir, 'model_best.pth.tar'))

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
    main()
