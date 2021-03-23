
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

##################################################### Training C_psi (classifier) ###########################################

np.random.seed(0)
torch.manual_seed(0)
CHECKPOINT_DIR = "../Models/"

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
FEATURE_DIM = 1024
IMAGE_SIZE = 256
CLASSES = 5
LR = 0.001

src_path = ''
target_path = ''

class FNet_VLCS(nn.Module):
 
  def __init__(self, hidden_layer_neurons, output_latent_dim):
    super(FNet_VLCS, self).__init__()
    self.alexnet_m = alexnet(pretrained=True, progress=False)

    self.alexnet_m.classifier[6] = nn.Linear(hidden_layer_neurons,  hidden_layer_neurons)
    self.fc1 = nn.Linear(hidden_layer_neurons, output_latent_dim)
   
  def forward(self, x):
    x = self.alexnet_m(x)
    x = self.fc1(x)
    return x

class DGdata(Dataset):
  def __init__(self, root_dir, image_size, domains=None, transform = None):
  
    self.root_dir = root_dir
    if root_dir[-1] != "/":
      self.root_dir = self.root_dir + "/"
    
    self.categories = ['bird', 'car', 'chair', 'dog', 'person']

    if domains is None:
      self.domains = ["caltech", "labelme", "pascal", "sun"]
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
    

vlcs_fnet = FNet_VLCS(4096, FEATURE_DIM)
checkpoint = torch.load('../Models/alex_fnet_vlcs.pt')
vlcs_fnet.load_state_dict(checkpoint['model_state_dict'])
vlcs_fnet = vlcs_fnet.to(dev)

layers = []
layers.append(nn.Linear(FEATURE_DIM, CLASSES))
classifier = torch.nn.Sequential(*layers).to(dev)
CELoss = nn.CrossEntropyLoss()
classifier = classifier.to(dev)

data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=IMAGE_SIZE), transforms.ToTensor()] )
ds = DGdata(".", IMAGE_SIZE, [src_path], transform=data_transforms)
dataloader = DataLoader(ds, batch_size=128, shuffle=True, num_workers = 4)

vlcs_fnet.eval()

opt = torch.optim.Adam(classifier.parameters(), lr=0.003)
for epoch in range(20):
  step_wise_loss = []
  step_wise_accuracy = []

  for image_batch, labels in (dataloader):
    image_batch = image_batch.float()
    if dev is not None:
      image_batch, labels = image_batch.to(dev), labels.to(dev)
        
    opt.zero_grad()
        
    z = vlcs_fnet(image_batch).to(dev)
    pred = classifier(z)
    loss = CELoss(pred, labels)
    accuracy = (pred.argmax(dim=1) == labels).float().sum()/pred.shape[0]
    loss.backward()
    opt.step()

    step_wise_loss.append(loss.detach().cpu().numpy())
    step_wise_accuracy.append(accuracy.detach().cpu().numpy())
  
  print("Epoch " + str(epoch) + " Loss " + str(np.mean(step_wise_loss)) + " Accuracy " + str(np.mean(step_wise_accuracy)))
    


############################################ 1-Nearest Neighbor ##############################################################


data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=IMAGE_SIZE), transforms.ToTensor()] )
test_data = DGdata(".", IMAGE_SIZE, [target_path], transform=data_transforms)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers = 4)

ds = DGdata(".", IMAGE_SIZE, [src_path], transform=data_transforms)
src_dataloader = DataLoader(ds, batch_size=1, shuffle=True, num_workers = 4)

discrete_fnet_space = []
for image, label in (src_data_dataloader):
  image = image.float().to(dev)
  h = vlcs_fnet(image)
  h = h.detach().view(image.size(0), -1).cpu().numpy()
  discrete_fnet_space.append(h)


step_wise_accuracy = []
neighbours = []
   
for image_batch, labels in (test_dataloader):
  image_batch = image_batch.float()
  if dev is not None:
    image_batch, labels = image_batch.to(dev), labels.to(dev)
        
  h = vlcs_fnet(image_batch)
  h = h.detach().view(image.size(0), -1).cpu().numpy()
  batches = int(len(image_batch)/1)
        
  for batch in (range(batches)):
      x_t = h[batch*1:(batch+1) * 1]
      lbl = labels[batch*1:(batch+1) * 1]
      no_1hot = lbl
      scores = []
      for x_s in discrete_fnet_space:
        x_s = x_s.reshape(1, FEATURE_DIM)
        x_t = x_t.reshape(1, FEATURE_DIM)
        x_s = x_s / np.linalg.norm(x_s)
        x_t = x_t / np.linalg.norm(x_t)
        x_s = x_s.reshape(FEATURE_DIM)
        x_t = x_t.reshape(FEATURE_DIM)
        cosine_score = np.dot(x_s,x_t)
        scores.append(cosine_score)
      index_max = np.argmax(scores)
      neighbours.append(discrete_fnet_space[index_max])
      feat = torch.from_numpy(discrete_fnet_space[index_max]).float().to(dev)
      pred = classifier(feat)
      accuracy = (pred.argmax(dim=1) == no_1hot).float().sum()/pred.shape[0]
      step_wise_accuracy.append(accuracy.detach().cpu().numpy())

print(np.mean(step_wise_accuracy))



