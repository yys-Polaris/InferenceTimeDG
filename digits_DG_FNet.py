
# coding: utf-8

# In[ ]:


from torch.utils.data import Dataset, DataLoader
import os
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, alexnet
import PIL
from torchlars import LARS
import cv2
import numpy as np

##################################################### Training f_theta network ###########################################


np.random.seed(0)
CHECKPOINT_DIR = "../Models/"

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 100
FEATURE_DIM = 256
IMAGE_SIZE = 32
CLASSES = 10
LR = 0.05

src_path = ''
target_path = ''

class DGdata(Dataset):
  def __init__(self, root_dir, image_size, domains=None, transform = None):
  
    self.root_dir = root_dir
    if root_dir[-1] != "/":
      self.root_dir = self.root_dir + "/"
    
    self.categories = ['0', '1', '2', '3', '4', '5', '6','7', '8', '9']

    if domains is None:
      self.domains = ["mnist", "mnist_m", "svhn", "syn"]
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


class GaussianBlur(object):
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    

class Backbone(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    @property
    def out_features(self):
        """Output feature dimension."""
        if self.__dict__.get('_out_features') is None:
            return None
        return self._out_features
    

class Convolution(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvNet(Backbone):

    def __init__(self, c_hidden=64):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)

        self._out_features = 2**2 * c_hidden

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32,             'Input to network must be 32x32, '             'but got {}x{}'.format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        return x.view(x.size(0), -1)
    

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

def training_loop(model, dataset, optimizer, tau=0.1, epochs=100, device=None):
  epoch_wise_loss = []
  epoch_wise_acc = []
  model.train()
  for epoch in (range(epochs)):
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
                  'loss': loss}, CHECKPOINT_DIR+"epoch_digits_resnet"+str(epoch)+".pt")
    epoch_wise_loss.append(np.mean(step_wise_loss))
    epoch_wise_acc.append(np.mean(step_wise_acc))
    print("epoch: {} loss: {:.3f} accuracy: {:.3f} ".format(epoch + 1, np.mean(step_wise_loss), np.mean(step_wise_acc)))

  return epoch_wise_loss, epoch_wise_acc, model

color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=IMAGE_SIZE),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(21)),
                                              transforms.ToTensor(),
                                              AddGaussianNoise(mean=0, std=0.2)] )
ds = DGdata(".", IMAGE_SIZE, [src_path], transform=data_transforms)
dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)

model = ConvNet(c_hidden=64)
model = model.to(dev)
optimizer = LARS(torch.optim.SGD(model.parameters(), lr=LR))
epoch_wise_loss, epoch_wise_acc, model = training_loop(model, dataloader, optimizer, tau=0.1, epochs=EPOCHS, device=dev)





