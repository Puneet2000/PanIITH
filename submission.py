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


class Data(Dataset):
	def __init__(self, csv_file, root_dir, transform=None,dim=224):
		self.image = pd.read_csv(csv_file)  # Reading solution.csv file
		self.imgname = self.image.iloc[:,0] # Reading indices of images
		self.category = self.image.iloc[:,1]
		self.root_dir = root_dir

		self.transform = transform          # Using transforms applied to dataset
		self.dim = dim

	def __len__(self):
		return len(self.image.index)

	def __getitem__(self, idx):
		img_name = self.root_dir + "/" + str(self.imgname[idx]) + ".png"
		image = cv2.imread(img_name)

		# Resizing Images, for VGG_13
		image = cv2.resize(image,(self.dim,self.dim))
		category = self.category[idx]
		if self.transform:
			image = self.transform(image) # Applying transforms
		return image,category

# Transforms applied to Training Images
transform = transforms.Compose([transforms.ToTensor()])

# Loading Test dataset
submissionset = Data(csv_file ='./sample.csv',root_dir='./testing',transform = transform)
submissionloader = DataLoader(dataset=submissionset, batch_size=2,shuffle=False, num_workers=0)

def create_submission_file(model_type='vgg'):
	if model_type == 'vgg':
		model = torch.load('./vgg.pt')
	else:
		model = torch.load('./alex.pt')

	# Pushing to cuda memory
	model = model.cuda()

	# Printing to console and further writing to csv
	print('id,category')
	i=1
	with torch.no_grad():
		for data in submissionloader:
		    inputs, labels= data
		    inputs = inputs.cuda()
		    labels = labels.cuda()
		    outputs = model(inputs)
		    outputs = F.softmax(outputs)
		    prob, predicted = torch.max(outputs.data,1)	# predicted scores and correspoding category
		    print(str(i)+','+str(int(predicted[0]+1)))	
		    print(str(i+1)+','+str(int(predicted[1]+1)))
		    i += 2

create_submission_file()
