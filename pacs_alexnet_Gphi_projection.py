
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

##################################################### Training G_phi & C_psi (classifier) ###########################################

np.random.seed(0)
torch.manual_seed(0)
CHECKPOINT_DIR = "../Models/"

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
FEATURE_DIM = 1024
IMAGE_SIZE = 224
CLASSES = 7
beta = 0.01
M = 20000
W = 5

src_path = ''
target_path = ''

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

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 1, 1)
    
class VAE_PACS_Alex(nn.Module):
    def __init__(self, image_channels=1, h_dim=256, z_dim=64):
        super(VAE_PACS_Alex, self).__init__()
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
            nn.ConvTranspose2d(h_dim, 16, kernel_size=4, stride=1),
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
        x = x.view(-1, 1, 32,32)
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z.view(-1, 1, 32,32)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
    

pacs_alex_fnet = FNet_Alex_PACS(4096, 1024)
checkpoint = torch.load('../Models/alex_fnet.pt')
pacs_alex_fnet.load_state_dict(checkpoint['model_state_dict'])
pacs_alex_fnet = pacs_alex_fnet.to(dev)

layers = []
layers.append(nn.Linear(FEATURE_DIM, CLASSES))
classifier = torch.nn.Sequential(*layers).to(dev)
CELoss = nn.CrossEntropyLoss()
classifier = classifier.to(dev)

data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=IMAGE_SIZE), transforms.ToTensor()] )
ds = DGdata(".", IMAGE_SIZE, [src_path], transform=data_transforms)
dataloader = DataLoader(ds, batch_size=64, shuffle=True, num_workers = 4)

pacs_alex_fnet.eval()

opt = torch.optim.Adam(classifier.parameters(), lr=0.003)
for epoch in range(30):
  step_wise_loss = []
  step_wise_accuracy = []

  for image_batch, labels in (dataloader):
    image_batch = image_batch.float()
    if dev is not None:
      image_batch, labels = image_batch.to(dev), labels.to(dev)
        
    # zero the parameter gradients
    opt.zero_grad()
        
    z = pacs_alex_fnet(image_batch).to(dev)
    pred = classifier(z)
    loss = CELoss(pred, labels)
    accuracy = (pred.argmax(dim=1) == labels).float().sum()/pred.shape[0]
    loss.backward()
    opt.step()

    step_wise_loss.append(loss.detach().cpu().numpy())
    step_wise_accuracy.append(accuracy.detach().cpu().numpy())
  
  print("Epoch " + str(epoch) + " Loss " + str(np.mean(step_wise_loss)) + " Accuracy " + str(np.mean(step_wise_accuracy)))
    

vae = VAE_PACS_Alex().to(dev)
VAEoptim = LARS(torch.optim.SGD(vae.parameters(), lr=0.005))
dataloader_vae = DataLoader(ds, batch_size=64, shuffle=True, num_workers = 4)
#modified loss
def loss_function(recon_x, x, mu, logvar):
    l2 = F.mse_loss(recon_x, x.view(-1, 1, 32, 32), reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    l1 = F.l1_loss(recon_x, x.view(-1, 1, 32, 32), reduction='mean')
    return l1 + l2 + KLD


def trainVAE(epoch):
    vae.train()
    train_loss = 0
    print(epoch)
    for batch_idx, (image_batch, _) in enumerate(dataloader_vae):
        image_batch = image_batch.float()
        image_batch = image_batch.to(dev)
        VAEoptim.zero_grad()
        h = pacs_alex_fnet(image_batch).to(dev)
        #print(h.shape)
        h = h.view(-1, 1, 32,32)
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
                  }, CHECKPOINT_DIR+"VAEepoch_pacs_alex_"+str(epoch)+".pt")
        
############################################ inference - target projection ##############################################################

vae.eval()

data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=IMAGE_SIZE), transforms.ToTensor()] )
test_data = DGdata(".", IMAGE_SIZE, [target_path], transform=data_transforms)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)


runs = 5
accuracy_per_run = []
for run in range(5):
  print('run:',run)
  step_wise_accuracy = []
  for image_batch, labels in (test_dataloader):
    image_batch = image_batch.float()
    if dev is not None:
      image_batch, labels = image_batch.to(dev), labels.to(dev)
        
    h = pacs_alex_fnet(image_batch).to(dev)
    h = h.detach()
    batches = int(len(image_batch)/1)
        
    for batch in (range(batches)):
        lbl = labels[batch*1:(batch+1) * 1]
        x_real = h[batch*1:(batch+1) * 1]
        no_1hot = lbl
        lbl = F.one_hot(lbl, CLASSES).float()
  
        zparam = torch.randn(1, 64).to(dev)
        zparam = zparam.detach().requires_grad_(True)
        zoptim = LARS(torch.optim.SGD([zparam], lr=beta,momentum=0.9, nesterov=True))
        Uparam = []
        L_s = []
        for itr in range(0, M):    ##projection
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



