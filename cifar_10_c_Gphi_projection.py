
# coding: utf-8

# In[ ]:

from torch.utils.data import Dataset, DataLoader
import os
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import PIL
from torchlars import LARS
import argparse
import shutil
import time
import math
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from torch.autograd import Variable
from torchlars import LARS
import cv2
import numpy as np

##################################################### Training G_phi & C_psi (classifier) ###########################################
np.random.seed(0)
torch.manual_seed(0)
CHECKPOINT_DIR = "../Models/"

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS              = 350

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
BATCH_SIZE = 64
beta = 0.01
M = 20000
W = 5

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
    

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input, size=64):
        return input.view(input.size(0), size, 1, 1)
    

class VAE_CIFAR_WResNet(nn.Module):
    def __init__(self, image_channels=1, h_dim=64, z_dim=32):
        super(VAE_CIFAR_WResNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 4, kernel_size=3, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2,padding=1),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        x = x.view(-1, 1, 16,16)
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z.view(-1, 1, 16,16)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

fwideResNet = WideResNet(LAYERS, FEATURE_DIM, WIDE,
                   droprate=DROPOUT,
                   use_bn=BATCHNORM,
                   use_fixup=FIXUP)

checkpoint = torch.load('../Models/cifar_10_fnet.pt')
fwideResNet.load_state_dict(checkpoint['model_state_dict'])
fwideResNet = fwideResNet.to(dev)

layers = []
layers.append(nn.Linear(FEATURE_DIM, CLASSES))
classifier = torch.nn.Sequential(*layers).to(dev)
CELoss = nn.CrossEntropyLoss()
classifier = classifier.to(dev)

normalize = transforms.Normalize(mean=[x / 255.0 for x in IMG_MEAN],
                                     std=[x / 255.0 for x in IMG_STD])

data_transforms = transforms.Compose([transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4,4,4,4), mode="reflect").squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

ds = DGdata(".", IMAGE_SIZE, [src_path], transform=data_transforms)
dataloader = DataLoader(ds, batch_size=64, shuffle=True, num_workers = 4)

fwideResNet.eval()

opt = torch.optim.Adam(classifier.parameters(), lr=0.003)
for epoch in range(20):
  step_wise_loss = []
  step_wise_accuracy = []

  for image_batch, labels in (dataloader):
    image_batch = image_batch.float()
    if dev is not None:
      image_batch, labels = image_batch.to(dev), labels.to(dev)
        
    # zero the parameter gradients
    opt.zero_grad()
        
    z = fwideResNet(image_batch).to(dev)
    pred = classifier(z)
    loss = CELoss(pred, labels)
    accuracy = (pred.argmax(dim=1) == labels).float().sum()/pred.shape[0]
    loss.backward()
    opt.step()

    step_wise_loss.append(loss.detach().cpu().numpy())
    step_wise_accuracy.append(accuracy.detach().cpu().numpy())
  
  print("Epoch " + str(epoch) + " Loss " + str(np.mean(step_wise_loss)) + " Accuracy " + str(np.mean(step_wise_accuracy)))
    

vae = VAE_CIFAR_WResNet().to(dev)
VAEoptim = LARS(torch.optim.SGD(vae.parameters(), lr=0.005))
dataloader_vae = DataLoader(ds, batch_size=64, shuffle=True, num_workers = 4)
#modified loss
def loss_function(recon_x, x, mu, logvar):
    l2 = F.mse_loss(recon_x, x.view(-1, 1, 16, 16), reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    l1 = F.l1_loss(recon_x, x.view(-1, 1, 16, 16), reduction='mean')
    return l1 + l2 + KLD


def trainVAE(epoch):
    vae.train()
    train_loss = 0
    print(epoch)
    for batch_idx, (image_batch, _) in enumerate(dataloader_vae):
        image_batch = image_batch.float()
        image_batch = image_batch.to(dev)
        VAEoptim.zero_grad()
        h = fwideResNet(image_batch).to(dev)
        #print(h.shape)
        h = h.view(-1, 1, 16,16)
        #print(h.shape)
        h=h.detach()
        recon_batch, mu, logvar = vae(h)
        loss = loss_function(recon_batch, h, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        VAEoptim.step()


    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(dataloader_vae.dataset)))

for epoch in range(1, 350):
    trainVAE(epoch)
    if (epoch)%10 == 0:
        torch.save({'epoch' : epoch,
                  'model_state_dict': vae.state_dict(),
                  'optimizer_state_dict': VAEoptim.state_dict()
                  }, CHECKPOINT_DIR+"VAEepoch_cifarC_wresnet_"+str(epoch)+".pt")
        
############################################ inference - target projection ##############################################################


vae.eval()
transform_test = transforms.Compose([transforms.ToTensor(),normalize])
test_data = DGdata(".", IMAGE_SIZE, [target_path], transform=transform_test)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)


runs = 5
elbow = ptr
accuracy_per_run = []
for run in range(5):
  print('run:',run)
  step_wise_accuracy = []
  for image_batch, labels in (test_dataloader):
    image_batch = image_batch.float()
    if dev is not None:
      image_batch, labels = image_batch.to(dev), labels.to(dev)
        
    h = fwideResNet(image_batch).to(dev)
    h = h.detach()
    batches = int(len(image_batch)/1)
        
    for batch in (range(batches)):
        lbl = labels[batch*1:(batch+1) * 1]
        x_real = h[batch*1:(batch+1) * 1]
        #print(x_real.shape)
        no_1hot = lbl
        lbl = F.one_hot(lbl, CLASSES).float()
  
        zparam = torch.randn(1, 32).to(dev)
        zparam = zparam.detach().requires_grad_(True)
        zoptim = LARS(torch.optim.SGD([zparam], lr=beta,momentum=0.9, nesterov=True))
        Uparam = []
        L_s = []
        
        for itr in range(0, M):          ## projection
          zoptim.zero_grad()
          xhat = vae.decode(zparam).to(dev)
          xhat = xhat.view(1, FEATURE_DIM)
          x_real = x_real.view(1, FEATURE_DIM)
          xhat = F.normalize(xhat, dim=1)
          x_real = F.normalize(x_real, dim=1)
          xhat = xhat.view(FEATURE_DIM)
          x_real = x_real.view(FEATURE_DIM)
          fnetloss = 1 - torch.dot(xhat,x_real)
          fnetloss.backward()
          zoptim.step()
          l = fnetloss.detach().cpu().numpy()
          u_param = zparam.detach().cpu().numpy()
          L_s.append(l)
          Uparam.append(u_param)
         
        L_s = np.asarray(L_s)
        Uparam = np.asarray(Uparam)
        smooth_L_s = np.cumsum(np.insert(L_s, 0, 0))
        s_vec = (smooth_L_s[W:] - smooth_L_s[:-W]) / W
        
        double_derivative=[]
        s_len=len(s_vec)
        for i in range(1,s_len-1):
            double_derivative.append(s_vec[i+1] + s_vec[i-1] - 2 * s_vec[i])
        double_derivative=np.asarray(double_derivative)
        
        zstar = torch.from_numpy(Uparam[np.argmax(double_derivative)])
        
        z_in = vae.decode(zstar.to(dev))
        z_in = z_in.view(-1, FEATURE_DIM)
        pred = classifier(z_in.to(dev))
        accuracy = (pred.argmax(dim=1) == no_1hot).float().sum()/pred.shape[0]
        step_wise_accuracy.append(accuracy.detach().cpu().numpy())

  print(np.mean(step_wise_accuracy))
  accuracy_per_run.append(np.mean(step_wise_accuracy))
print(np.mean(accuracy_per_run))





