
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
FEATURE_DIM = 256
IMAGE_SIZE = 32
CLASSES = 10
beta = 0.01
M = 20000
W = 5

src_path = ''
target_path = ''

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

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input, size=64):
        return input.view(input.size(0), size, 1, 1)
    

class VAE_Digits(nn.Module):
    def __init__(self, image_channels=1, h_dim=64, z_dim=32):
        super(VAE_Digits, self).__init__()
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

digits_fnet = ConvNet(c_hidden=64)
checkpoint = torch.load('../Models/digits_fnet.pt')
digits_fnet.load_state_dict(checkpoint['model_state_dict'])
digits_fnet = digits_fnet.to(dev)

layers = []
layers.append(nn.Linear(FEATURE_DIM, CLASSES))
classifier = torch.nn.Sequential(*layers).to(dev)
CELoss = nn.CrossEntropyLoss()
classifier = classifier.to(dev)

data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=IMAGE_SIZE), transforms.ToTensor()] )
ds = DGdata(".", IMAGE_SIZE, [src_path], transform=data_transforms)
dataloader = DataLoader(ds, batch_size=64, shuffle=True, num_workers = 4)

digits_fnet.eval()

opt = torch.optim.Adam(classifier.parameters(), lr=0.003)
for epoch in range(15):
  step_wise_loss = []
  step_wise_accuracy = []

  for image_batch, labels in (dataloader):
    image_batch = image_batch.float()
    if dev is not None:
      image_batch, labels = image_batch.to(dev), labels.to(dev)
        
    # zero the parameter gradients
    opt.zero_grad()
        
    z = digits_fnet(image_batch).to(dev)
    pred = classifier(z)
    loss = CELoss(pred, labels)
    accuracy = (pred.argmax(dim=1) == labels).float().sum()/pred.shape[0]
    loss.backward()
    opt.step()

    step_wise_loss.append(loss.detach().cpu().numpy())
    step_wise_accuracy.append(accuracy.detach().cpu().numpy())
  
  print("Epoch " + str(epoch) + " Loss " + str(np.mean(step_wise_loss)) + " Accuracy " + str(np.mean(step_wise_accuracy)))
    

vae = VAE_Digits().to(dev)
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
        h = digits_fnet(image_batch).to(dev)
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

for epoch in range(1, 150):
    trainVAE(epoch)
    if (epoch)%10 == 0:
        torch.save({'epoch' : epoch,
                  'model_state_dict': vae.state_dict(),
                  'optimizer_state_dict': VAEoptim.state_dict()
                  }, CHECKPOINT_DIR+"VAEepoch_digits_"+str(epoch)+".pt")
        
############################################ inference - target projection ##############################################################

vae.eval()

data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=IMAGE_SIZE), transforms.ToTensor()] )
test_data = DGdata(".", IMAGE_SIZE, [target_path], transform=data_transforms)
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
        
    h = digits_fnet(image_batch).to(dev)
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
        for itr in range(0, M):      ## projection
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





