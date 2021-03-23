LR                  = 0.1
BATCH_SIZE          = 128
EPOCHS              = 200

LAYERS              = 16        # total number of layers
WIDE                = 4        # widen factor
BATCHNORM           = True      # apply BatchNorm
FIXUP               = True      # apply Fixup
DROPOUT             = 0.3         # dropout probability (default: 0.0)

AUGMENT             = True      # use standard augmentation (default: True)

# Image Setup
CLASSES             = 10
IMAGE_SIZE         = 32
IMG_CHANNELS       = 3
IMG_MEAN           = [125.3, 123.0, 113.9]
IMG_STD            = [63.0, 62.1, 66.7]

FEATURE_DIM = 256

# Setup SGD
momentum = 0.9
nesterov = True
weight_decay = 5e-4
start_epoch = 0

print_freq = 10


import argparse
import os
import shutil
import time
import math

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import PIL
from torchlars import LARS
import cv2
import numpy as np

##################################################### Training f_theta network ###########################################

np.random.seed(0)
CHECKPOINT_DIR = "../Models/"

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

src_path = ''
target_path = ''

class DGdata(Dataset):
  def __init__(self, root_dir, image_size, domains=None, transform = None):
  
    self.root_dir = root_dir
    if root_dir[-1] != "/":
      self.root_dir = self.root_dir + "/"
    
    self.categories = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']

    if domains is None:
      self.domains = ["cifar"]
    else:
      self.domains = domains
    
    if transform is None:
      self.transform = transforms.ToTensor()
    else:
      self.transform = transform
    # make a list of all the files in the root_dir
    # and read the labels
    self.img_files = []
    self.labels = []
    self.domain_labels = []
    for domain in self.domains:
      for category in self.categories:
        for image in os.listdir(self.root_dir+domain+'/'+category):
          self.img_files.append(image)
          self.labels.append(self.categories.index(category))
          self.domain_labels.append(self.domains.index(domain))
  
  def __len__(self):
    return len(self.img_files)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    img_path = self.root_dir + self.domains[self.domain_labels[idx]] + "/" + self.categories[self.labels[idx]] + "/" + self.img_files[idx]
    
    image = PIL.Image.open(img_path)
    label = self.labels[idx]

    return self.transform(image), label


class BasicBlock(nn.Module):
    droprate = 0.0
    use_bn = True
    use_fixup = False
    fixup_l = 12

    def __init__(self, in_planes, out_planes, stride):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.equalInOut = in_planes == out_planes
        self.conv_res = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_res = not self.equalInOut and self.conv_res or None

        assert self.use_fixup or self.use_bn, "Need to use at least one thing: Fixup or BatchNorm"

        if self.use_fixup:
            self.multiplicator = nn.Parameter(torch.ones(1,1,1,1))
            self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1,1,1,1))] * 4)

            k = self.conv1.kernel_size[0] * self.conv1.kernel_size[1] * self.conv1.out_channels
            self.conv1.weight.data.normal_(0, self.fixup_l ** (-0.5) * math.sqrt(2. / k))
            self.conv2.weight.data.zero_()
            
            if self.conv_res is not None:
                k = self.conv_res.kernel_size[0] * self.conv_res.kernel_size[1] * self.conv_res.out_channels
                self.conv_res.weight.data.normal_(0, math.sqrt(2. / k))

    def forward(self, x):
        if self.use_bn:
            x_out = self.relu(self.bn1(x))
            out = self.relu(self.bn2(self.conv1(x_out)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.conv2(out)
        else:
            x_out = self.relu(x + self.biases[0])
            out = self.conv1(x_out) + self.biases[1]
            out = self.relu(out) + self.biases[2]
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.multiplicator * self.conv2(out) + self.biases[3]

        if self.equalInOut:
            return torch.add(x, out)

        return torch.add(self.conv_res(x_out), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride):
        layers = []

        for i in range(int(nb_layers)):
            _in_planes = i == 0 and in_planes or out_planes
            _stride = i == 0 and stride or 1
            layers.append(block(_in_planes, out_planes, _stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

#@registry.Model
class WideResNet(nn.Module):
    def __init__(self, depth, feat_dim, widen_factor=1, droprate=0.0, use_bn=True, use_fixup=False):
        super(WideResNet, self).__init__()

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        assert (depth - 4) % 6 == 0, "You need to change the number of layers"
        n = (depth - 4) / 6

        BasicBlock.droprate = droprate
        BasicBlock.use_bn = use_bn
        BasicBlock.fixup_l = n * 3
        BasicBlock.use_fixup = use_fixup
        block = BasicBlock
        
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2)
        
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], feat_dim)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / k))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                if use_fixup:
                    m.weight.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 5 at 50th, 100th and 150th epochs"""
    lr_tmp = LR * ((0.2 ** int(epoch >= 50)) * (0.2 ** int(epoch >= 100))* (0.2 ** int(epoch >= 150)))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_tmp
        

def train_step(x, labels, model, optimizer, tau):
  optimizer.zero_grad()
  # Forward pass
  z = model(x)

  # Calculate loss
  z = F.normalize(z, dim=1)
  pairwise_labels = torch.flatten(torch.matmul(labels, labels.t()))
  logits = torch.flatten(torch.matmul(z, z.t())) / tau
  loss = F.binary_cross_entropy_with_logits(logits, pairwise_labels)
  pred = torch.sigmoid(logits)   # whether two images are similar or not
  accuracy = (pred.round().float() == pairwise_labels).sum()/float(pred.shape[0])
      
  # Perform train step
  #optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()

def training_loop(model, dataset, optimizer, tau=0.1, epochs=200, device=None):
  epoch_wise_loss = []
  epoch_wise_acc = []
  model.train()
  for epoch in (range(epochs)):
    adjust_learning_rate(optimizer, epoch+1)
    step_wise_loss = []
    step_wise_acc = []
    for image_batch, labels in (dataset):
      image_batch = image_batch.float()
      if dev is not None:
        image_batch, labels = image_batch.to(device), labels.to(device)
      labels_onehot = F.one_hot(labels, CLASSES).float()
      loss, accuracy = train_step(image_batch, labels_onehot, model, optimizer, tau)
      step_wise_loss.append(loss)
      step_wise_acc.append(accuracy)


    if (epoch+1)%20 == 0:
      torch.save({'epoch' : epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss}, CHECKPOINT_DIR+"epoch_cifarC_wideresnet"+str(epoch)+".pt")
    epoch_wise_loss.append(np.mean(step_wise_loss))
    epoch_wise_acc.append(np.mean(step_wise_acc))
    print("epoch: {} loss: {:.3f} accuracy: {:.3f} ".format(epoch + 1, np.mean(step_wise_loss), np.mean(step_wise_acc)))

  return epoch_wise_loss, epoch_wise_acc, model


normalize = transforms.Normalize(mean=[x / 255.0 for x in IMG_MEAN],
                                     std=[x / 255.0 for x in IMG_STD])
    
    # augmentation
data_transforms = transforms.Compose([transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4,4,4,4), mode="reflect").squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

ds = DGdata(".", IMAGE_SIZE, [src_path], transform=data_transforms)
dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)

fwideResNet = WideResNet(LAYERS, FEATURE_DIM, WIDE,
                   droprate=DROPOUT,
                   use_bn=BATCHNORM,
                   use_fixup=FIXUP)

fwideResNet = fwideResNet.to(dev)
optimizer = LARS(torch.optim.SGD(fwideResNet.parameters(), lr=LR))
epoch_wise_loss, epoch_wise_acc, model = training_loop(model, dataloader, optimizer, tau=0.1, epochs=EPOCHS, device=dev)
