
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
IMAGE_SIZE = 256
CLASSES = 5
ngpu = 4
nz = 64
ngf = 16
nc = 1
ndf = 16
lr = 0.0002
beta1 = 0.5
num_epochs = 450
beta = 0.01
M = 20000
W = 5

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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 1, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    
    
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
    
# Create the generator
netG = Generator(ngpu).to(dev)

# Handle multi-gpu if desired
if (dev.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)


# Create the Discriminator
netD = Discriminator(ngpu).to(dev)

# Handle multi-gpu if desired
if (dev.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)

    
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
dataloader = DataLoader(ds, batch_size=64, shuffle=True, num_workers = 4)

vlcs_fnet.eval()

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
        
    z = vlcs_fnet(image_batch).to(dev)
    pred = classifier(z)
    loss = CELoss(pred, labels)
    accuracy = (pred.argmax(dim=1) == labels).float().sum()/pred.shape[0]
    loss.backward()
    opt.step()

    step_wise_loss.append(loss.detach().cpu().numpy())
    step_wise_accuracy.append(accuracy.detach().cpu().numpy())
  
  print("Epoch " + str(epoch) + " Loss " + str(np.mean(step_wise_loss)) + " Accuracy " + str(np.mean(step_wise_accuracy)))
    

# Initialize BCELoss function
criterion = nn.BCELoss()

fixed_noise = torch.randn(64, nz, 1, 1, device=dev)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(dev)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=dev)
        # Forward pass real batch through D
        h = vlcs_fnet(real_cpu).to(dev)
        h = h.view(-1, 1, 32,32)
        output = netD(h).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=dev)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            torch.save({'epoch' : epoch,
                  'model_state_dict': netG.state_dict(),
                  'optimizer_state_dict': optimizerG.state_dict()},
                  CHECKPOINT_DIR+"GAN_vlcs_alex_"+str(epoch)+".pt")

        
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        iters += 1
        
############################################ inference - target projection ##############################################################

netG.eval()

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
        
    h = vlcs_fnet(image_batch).to(dev)
    h = h.detach()
    batches = int(len(image_batch)/1)
        
    for batch in (range(batches)):
        lbl = labels[batch*1:(batch+1) * 1]
        x_real = h[batch*1:(batch+1) * 1]
        #print(x_real.shape)
        no_1hot = lbl
        lbl = F.one_hot(lbl, CLASSES).float()
  
        zparam = torch.randn(1, 64).to(dev)
        zparam = zparam.view(-1, 64, 1, 1)
        zparam = zparam.detach().requires_grad_(True)
        zoptim = LARS(torch.optim.SGD([zparam], lr=beta,momentum=0.9, nesterov=True))
        Uparam = []
        L_s = []
        for itr in range(0, M):        ##projection
          zoptim.zero_grad()
          xhat = netG(zparam).to(dev)
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
        
        z_in = netG(zstar.to(dev))
        z_in = z_in.view(-1, FEATURE_DIM)
        pred = classifier(z_in.to(dev))
        accuracy = (pred.argmax(dim=1) == no_1hot).float().sum()/pred.shape[0]
        step_wise_accuracy.append(accuracy.detach().cpu().numpy())

  print(np.mean(step_wise_accuracy))
  accuracy_per_run.append(np.mean(step_wise_accuracy))
print(np.mean(accuracy_per_run))







