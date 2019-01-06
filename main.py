import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import os
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torchvision.utils import save_image
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models


# Dataset Loader class
class Data(Dataset):
	def __init__(self, csv_file, root_dir, transform=None, dim=224):
		self.image = pd.read_csv(csv_file)  # Reading solution.csv file
		self.imgname = self.image.iloc[:,0] # Reading indices of images
		self.category = self.image.iloc[:,1]
		self.root_dir = root_dir
		self.transform = transform 			# Applying transforms to dataset
		self.dim = dim

	def __len__(self):
		return len(self.image.index)

	def __getitem__(self, idx):
		# Reading image
		img_name = self.root_dir + "/" + str(self.imgname[idx]) + ".png"
		image = cv2.imread(img_name)

		# Resizing Images
		image = cv2.resize(image,(self.dim,self.dim))
		category = self.category[idx]
		if self.transform:
			image = self.transform(image) # Applying transforms
		return image,category

# Transforms applied to Training Images
transform = transforms.Compose([
								transforms.ToTensor(),      
								transforms.ToPILImage(),    
								transforms.RandomRotation(60, resample=False, expand=False, center=None),
								transforms.RandomHorizontalFlip(),
								transforms.RandomVerticalFlip(),
								transforms.ToTensor()
								])	

# Loading the training dataset
trainset = Data(csv_file='./solution.csv',root_dir='./training',transform = transform)

# Splitting data into test and train
train_dataset, test_dataset = torch.utils.data.random_split(trainset, [4000, 1000])

# Train Loader and Test Loader
trainloader = DataLoader(dataset=train_dataset, batch_size=20,shuffle=True, num_workers=4)
testloader = DataLoader(dataset=test_dataset, batch_size=20,shuffle=False, num_workers=4)

def train(model_type='vgg'):
	# Loading pretrined VGG_13 model from ->
    #        'vgg13_bn': https://download.pytorch.org/models/vgg13_bn-abd245e5.pth
    #  		 'alexnet' : https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
	if model_type == 'vgg':
		model = models.vgg13_bn(True)
		# Classifier (fully connected) layers of network for final classification
		model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),   # FC layer
            nn.ReLU(True),                  # Rectified Linear Units activation funcitons
            nn.Dropout(),                   # Applying dropout to (default value 0.5)
            nn.Linear(4096, 4096),          # Another FC layer for transformation
            nn.ReLU(True),                  # Rectified Layer for activation   
            nn.Dropout(),                   # Dropout with deafult value 0.5
            nn.Linear(4096, 6),)             # Final classification layer

	else:
		model = models.alexnet(True)
		# Classifier (fully connected) layers of network for final classification
		model.classifier  = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 6),)

	# Pushing to cuda memory
	model = model.cuda()

	# Cross Entropy Loss as loss function
	criterion = nn.CrossEntropyLoss()
	# SGD as optimizer, with learning rate decayed based on loss and convergence factors
	optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.9)

	# Running for 100 epochs, seems to converger better
	for epoch in range(100):
		ave_loss = []    # Average loss over all epochs
		for i,(inputs,labels)in enumerate(trainloader):
	        # Loading inputs
			inputs, labels = inputs.cuda(), labels.cuda()
			optimizer.zero_grad()   # Nullifying the gradients of last iterations 
			output = model(inputs)  # Fetching outputs
			loss = criterion(output,labels-1)    
			loss.backward()   # Calculating gradients
			optimizer.step()  # Updating values based on gradients
			ave_loss.append(loss.item())
			if i%50 == 49:
				if model_type == 'vgg':
					torch.save(model,'./vgg.pt')     # Saving after every 50 iterations
				else:
					torch.save(model,'./alex.pt')
		print('epoch {0} , loss {1}'.format(epoch+1,np.mean(ave_loss)))

# Tarining the model
train('vgg')

# Evaluating the model
def evaluate(model_type='vgg'):	
	if model_type == 'vgg':
		model = torch.load('./vgg.pt')
	else:
		model = torch.load('./alex.pt')
	# Evaluating model on test set created form the given dataset
	correct, total = 0, 0
	with torch.no_grad():
		for i, (inputs, labels) in enumerate(testloader):
			inputs,labels = inputs.cuda(),labels.cuda()
			outputs = model(inputs)   # Evaluating model
			_, predicted = torch.max(outputs.data,1)
			total += labels.size(0)
			correct += (predicted == labels-1).sum().item()

	print('Accuracy of the network on the  test images: %0.6f %%' % (
	    100 * correct / total))

evaluate()