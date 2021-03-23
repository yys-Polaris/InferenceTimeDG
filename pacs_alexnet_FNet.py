
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
EPOCHS = 250
FEATURE_DIM = 1024
IMAGE_SIZE = 224
CLASSES = 7
LR = 0.001

src_path = ''
target_path = ''

class DGdata(Dataset):
  def __init__(self, root_dir, image_size, domains=None, transform = None):
  
    self.root_dir = root_dir
    if root_dir[-1] != "/":
      self.root_dir = self.root_dir + "/"
    
    self.categories = ['giraffe', 'horse', 'guitar', 'person', 'dog', 'house', 'elephant']

    if domains is None:
      self.domains = ["photo", "sketch", "art_painting", "cartoon"]
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
    

class FNet_Alex_PACS(nn.Module):
 
  def __init__(self, hidden_layer_neurons, output_latent_dim):
    super(FNet_Alex_PACS, self).__init__()
    self.alexnet_m = alexnet(pretrained=True, progress=False)

    self.alexnet_m.classifier[6] = nn.Linear(hidden_layer_neurons,  hidden_layer_neurons)
    self.fc1 = nn.Linear(hidden_layer_neurons, output_latent_dim)
   
  def forward(self, x):
    x = self.alexnet_m(x)
    x = self.fc1(x)
    return x
    

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

def training_loop(model, dataset, optimizer, tau=0.1, epochs=250, device=None):
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
                  'loss': loss}, CHECKPOINT_DIR+"epoch_pacs_alex"+str(epoch)+".pt")
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

model = FNet_Alex_PACS(4096, FEATURE_DIM)
model = model.to(dev)
optimizer = LARS(torch.optim.SGD(model.parameters(), lr=LR))
epoch_wise_loss, epoch_wise_acc, model = training_loop(model, dataloader, optimizer, tau=0.1, epochs=EPOCHS, device=dev)


