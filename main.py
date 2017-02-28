from __future__ import print_function
import os
import time
import random
import argparse
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

### load project files
import models
from models import weights_init

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outDir', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--model', type=int, default=1, help='1 for dcgan, 2 for illustrationGAN-like-GAN')
parser.add_argument('--d_labelSmooth', type=float, default=0, help='for D, use soft label "1-labelSmooth" for real samples')
parser.add_argument('--n_extra_layers_d', type=int, default=0, help='number of extra conv layers in D')
parser.add_argument('--n_extra_layers_g', type=int, default=1, help='number of extra conv layers in G')
parser.add_argument('--binary', action='store_true', help='z from bernoulli distribution, with prob=0.5')

# simply prefer this way
# arg_list = [
#     '--dataRoot', '/home/jielei/data/danbooru-faces',
#     '--workers', '12',
#     '--batchSize', '128',
#     '--imageSize', '64',
#     '--nz', '100',
#     '--ngf', '64',
#     '--ndf', '64',
#     '--niter', '80',
#     '--lr', '0.0002',
#     '--beta1', '0.5',
#     '--cuda', 
#     '--ngpu', '1',
#     '--netG', '',
#     '--netD', '',
#     '--outDir', './results',
#     '--model', '1',
#     '--d_labelSmooth', '0.1', # 0.25 from imporved-GAN paper 
#     '--n_extra_layers_d', '0',
#     '--n_extra_layers_g', '1', # in the sense that generator should be more powerful
# ]

args = parser.parse_args()
# opt = parser.parse_args(arg_list)
print(opt)

try:
    os.makedirs(opt.outDir)
except OSError:
    pass

opt.manualSeed = random.randint(1,10000) # fix seed, a scalar
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
nc = 3
ngpu = opt.ngpu
nz = opt.nz
ngf = opt.ngf
ndf = opt.ndf
n_extra_d = opt.n_extra_layers_d
n_extra_g = opt.n_extra_layers_g

dataset = dset.ImageFolder(
    root=opt.dataRoot,
    transform=transforms.Compose([
            transforms.Scale(opt.imageSize),
            # transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), # bring images to (-1,1)
        ])
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=opt.workers)

# load models 
if opt.model == 1:
    netG = models._netG_1(ngpu, nz, nc, ngf, n_extra_g)
    netD = models._netD_1(ngpu, nz, nc, ndf, n_extra_d)
elif opt.model == 2:
    netG = models._netG_2(ngpu, nz, nc, ngf)
    netD = models._netD_2(ngpu, nz, nc, ndf)

netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()
criterion_MSE = nn.MSELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
if opt.binary:
    bernoulli_prob = torch.FloatTensor(opt.batchSize, nz, 1, 1).fill_(0.5)
    fixed_noise = torch.bernoulli(bernoulli_prob)
else:
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterion_MSE.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    
input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        start_iter = time.time()
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        label.data.resize_(batch_size).fill_(real_label - opt.d_labelSmooth) # use smooth label for discriminator

        output = netD(input)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()
        # train with fake
        noise.data.resize_(batch_size, nz, 1, 1)
        if opt.binary:
            bernoulli_prob.resize_(noise.data.size())
            noise.data.copy_(2*(torch.bernoulli(bernoulli_prob)-0.5))
        else:
            noise.data.normal_(0, 1)
        fake,z_prediction = netG(noise)
        label.data.fill_(fake_label)
        output = netD(fake.detach()) # add ".detach()" to avoid backprop through G
        errD_fake = criterion(output, label)
        errD_fake.backward() # gradients for fake/real will be accumulated
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step() # .step() can be called once the gradients are computed

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.data.fill_(real_label) # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward(retain_variables=True) # True if backward through the graph for the second time
        if opt.model == 2: # with z predictor
            errG_z = criterion_MSE(z_prediction, noise)
            errG_z.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()
        
        end_iter = time.time()
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2, end_iter-start_iter))
        if i % 100 == 0:
            # the first 64 samples from the mini-batch are saved.
            vutils.save_image(real_cpu[0:64,:,:,:],
                    '%s/real_samples.png' % opt.outDir, nrow=8)
            fake,_ = netG(fixed_noise)
            vutils.save_image(fake.data[0:64,:,:,:],
                    '%s/fake_samples_epoch_%03d.png' % (opt.outDir, epoch), nrow=8)
    if epoch % 1 == 0:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outDir, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outDir, epoch))
