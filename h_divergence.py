
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
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

##################################################### Training Deep All, C_psi (classifier) ###########################################

np.random.seed(0)
torch.manual_seed(0)
CHECKPOINT_DIR = "../Models/"

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
FEATURE_DIM = 256
IMAGE_SIZE = 224
CLASSES = 7

src_path = ''
target_path = ''

class FNet_PACS_ResNet(nn.Module):
 
  def __init__(self, hidden_layer_neurons, output_latent_dim):
    super(FNet_PACS_ResNet, self).__init__()
    resnet = resnet18(pretrained=True, progress=False)
    
    self.resnet = nn.Sequential(*list(resnet.children())[:-1])
    self.fc1 = nn.Linear(resnet.fc.in_features,  hidden_layer_neurons)
    self.fc2 = nn.Linear(hidden_layer_neurons, output_latent_dim)
   
  def forward(self, x):
    x = self.resnet(x)
    x = x.squeeze()

    x = self.fc1(x)
    x = F.leaky_relu(x, negative_slope=0.2)

    x = self.fc2(x)
    return x

class DeepAll(nn.Module):
 
  def __init__(self, hidden_layer_neurons, output_latent_dim):
    super(DeepAll, self).__init__()
    resnet = resnet18(pretrained=True, progress=False)
    
    self.resnet = nn.Sequential(*list(resnet.children())[:-1])
    self.fc1 = nn.Linear(resnet.fc.in_features,  hidden_layer_neurons)
    self.fc2 = nn.Linear(hidden_layer_neurons, output_latent_dim)
    self.fc3 = nn.Linear(output_latent_dim, CLASSES)
   
  def forward(self, x):
    x = self.resnet(x)
    x = x.squeeze()

    x = self.fc1(x)
    x = F.leaky_relu(x, negative_slope=0.2)

    x = self.fc2(x)
    x = F.leaky_relu(x, negative_slope=0.2)

    x = self.fc3(x)
    return x

class DeepAll_Feat(nn.Module):
 
  def __init__(self, hidden_layer_neurons, output_latent_dim):
    super(DeepAll_Feat, self).__init__()
    resnet = resnet18(pretrained=True, progress=False)
    
    self.resnet = nn.Sequential(*list(resnet.children())[:-1])
    self.fc1 = nn.Linear(resnet.fc.in_features,  hidden_layer_neurons)
    self.fc2 = nn.Linear(hidden_layer_neurons, output_latent_dim)
    self.fc3 = nn.Linear(output_latent_dim, 7)
   
  def forward(self, x):
    x = self.resnet(x)
    x = x.squeeze()

    x = self.fc1(x)
    x = F.leaky_relu(x, negative_slope=0.2)

    feat = self.fc2(x)
    x = F.leaky_relu(itmd, negative_slope=0.2)

    x = self.fc3(x)
    return feat

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


def train_step(x, labels, model, optimizer, CELoss):
  optimizer.zero_grad()
  # Forward pass
  pred = model(x)
  # Calculate loss

  loss = CELoss(pred, labels)  #without Fnet
  accuracy = (pred.argmax(dim=1) == labels).float().sum()/pred.shape[0]
  loss.backward()
  optimizer.step()

  return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()


def training_loop(model, dataset, optimizer, CELoss, epochs=200, device=None):
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
      #print(labels_onehot.shape)
      loss, accuracy = train_step(image_batch, labels, model, optimizer, CELoss)
      step_wise_loss.append(loss)
      step_wise_acc.append(accuracy)


    if (epoch+1)%10 == 0:
      torch.save({'epoch' : epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss}, CHECKPOINT_DIR+"pacs_res_deepall"+str(epoch)+".pt")
    epoch_wise_loss.append(np.mean(step_wise_loss))
    epoch_wise_acc.append(np.mean(step_wise_acc))
    print("epoch: {} loss: {:.3f} accuracy: {:.3f} ".format(epoch + 1, np.mean(step_wise_loss), np.mean(step_wise_acc)))

  return epoch_wise_loss, epoch_wise_acc, model


pacs_resnet_fnet = FNet_PACS_ResNet(512, FEATURE_DIM)   #Fnet trained on P, A, C as source
checkpoint = torch.load('../Models/resnet_fnet.pt')
pacs_resnet_fnet.load_state_dict(checkpoint['model_state_dict'])
pacs_resnet_fnet = pacs_resnet_fnet.to(dev)

layers = []
layers.append(nn.Linear(FEATURE_DIM, CLASSES))
classifier = torch.nn.Sequential(*layers).to(dev)
CELoss = nn.CrossEntropyLoss()
classifier = classifier.to(dev)

data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=IMAGE_SIZE), transforms.ToTensor()] )
ds = DGdata(".", IMAGE_SIZE, [src_path], transform=data_transforms)
dataloader = DataLoader(ds, batch_size=128, shuffle=True, num_workers = 4)


pacs_resnet_fnet.eval()

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
        
    z = pacs_resnet_fnet(image_batch).to(dev)
    pred = classifier(z)
    loss = CELoss(pred, labels)
    accuracy = (pred.argmax(dim=1) == labels).float().sum()/pred.shape[0]
    loss.backward()
    opt.step()

    step_wise_loss.append(loss.detach().cpu().numpy())
    step_wise_accuracy.append(accuracy.detach().cpu().numpy())
  
  print("Epoch " + str(epoch) + " Loss " + str(np.mean(step_wise_loss)) + " Accuracy " + str(np.mean(step_wise_accuracy)))
    

pacs_res_deepall = DeepAll(512, FEATURE_DIM)   #without fnet (only Deep All)
pacs_res_deepall = pacs_res_deepall.to(dev)
optimizer = LARS(torch.optim.SGD(pacs_res_deepall.parameters(), lr=0.001))
CELoss = nn.CrossEntropyLoss()
epoch_wise_loss, epoch_wise_acc, pacs_res_deepall = training_loop(pacs_res_deepall, dataloader, optimizer, CELoss, epochs=200, device=dev)


pacs_res_deepall = DeepAll_Feat(512, FEATURE_DIM)  # load DeepAll model
checkpoint = torch.load('../Models/pacs_res_deepall.pt')
pacs_res_deepall.load_state_dict(checkpoint['model_state_dict'])
pacs_res_deepall = pacs_res_deepall.to(dev)
pacs_res_deepall.eval()

################################################  H-divergence #################################################

BATCH_SIZE = 500  #A-distance calculated between 500 random samples from each domain
domains = ["photo", "sketch", "art_painting", "cartoon"]
data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=IMAGE_SIZE),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])] )
ds1 = DGdata(".", IMAGE_SIZE, ['sketch'], transform=data_transforms)
dataloader1 = DataLoader(ds1, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
ds2 = DGdata(".", IMAGE_SIZE, ['photo'], transform=data_transforms)
dataloader2 = DataLoader(ds2, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
ds3 = DGdata(".", IMAGE_SIZE, ['art_painting'], transform=data_transforms)
dataloader3 = DataLoader(ds3, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
ds4 = DGdata(".", IMAGE_SIZE, ['cartoon'], transform=data_transforms)
dataloader4 = DataLoader(ds4, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)

data1, labels1 = next(iter(dataloader1))
domain1 = torch.zeros_like(labels1)
data2, labels2 = next(iter(dataloader2))
domain2 = torch.ones_like(labels2)
data3, labels3 = next(iter(dataloader3))
domain3 = 2*torch.ones_like(labels3)
data4, labels4 = next(iter(dataloader4))
domain4 = 3*torch.ones_like(labels4)

h_div_matrix_deepall = np.zeros([len(domains), len(domains)])
all_data = [data1, data2, data3, data4]
labels = [domain1, domain2, domain3, domain4]

for d1, name1 in enumerate(domains):
    for d2, name2 in enumerate(domains):
        if d1 != d2:
            print('Domain 1', name1)
            print('Domain 2', name2)

            data_d1 = all_data[d1]
            label_d1 = labels[d1].numpy()
            data_d2 = all_data[d2]
            label_d2 = labels[d2].numpy()

            features_d1_deepall = pacs_res_deepall(data_d1).detach().view(data_d1.size(0), -1).numpy()
            features_d2_deepall = pacs_res_deepall(data_d2).detach().view(data_d2.size(0), -1).numpy()
            x = np.vstack((features_d1_deepall, features_d2_deepall))
            y = np.hstack((label_d1, label_d2))
            x, y = shuffle(x, y)
        
            model = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1, kernel='linear'))
            acc = cross_val_score(model, x, y.ravel(), cv=5, scoring='accuracy')

            print('Accuracy:', acc)
            h_div_matrix_deepall[d1, d2] =     2*(1.-2.*(1-np.mean(acc)))
            
print(h_div_matrix_deepall)


h_div_matrix_fnet = np.zeros([len(domains), len(domains)])

for d1, name1 in enumerate(domains):
    for d2, name2 in enumerate(domains):
        if d1 != d2:
            print('Domain 1', name1)
            print('Domain 2', name2)

            data_d1 = all_data[d1]
            label_d1 = labels[d1].numpy()
            data_d2 = all_data[d2]
            label_d2 = labels[d2].numpy()

            features_d1_fnet = pacs_resnet_fnet(data_d1).detach().view(data_d1.size(0), -1).numpy()
            features_d2_fnet = pacs_resnet_fnet(data_d2).detach().view(data_d2.size(0), -1).numpy()
            x = np.vstack((features_d1_fnet, features_d2_fnet))
            y = np.hstack((label_d1, label_d2))
            x, y = shuffle(x, y)
        
            model = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1, kernel='linear'))
            acc = cross_val_score(model, x, y.ravel(), cv=5, scoring='accuracy')

            print('Accuracy:', acc)
            h_div_matrix_fnet[d1, d2] =     2*(1.-2.*(1-np.mean(acc)))
            
print(h_div_matrix_fnet)



