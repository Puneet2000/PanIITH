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



class PanIITDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None,dim=224,grayscale=False):
        self.image = pd.read_csv(csv_file)
        self.imgname = self.image.iloc[:,0]
        self.category = self.image.iloc[:,1]
        self.root_dir = root_dir
        self.transform = transform
        self.dim = dim
        self.grayscale = grayscale

    def __len__(self):
        return len(self.image.index)

    def __getitem__(self, idx):
    	img_name = self.root_dir + "/" + str(self.imgname[idx]) + ".png"
    	if self.grayscale:
    		image = cv2.imread(img_name , cv2.IMREAD_GRAYSCALE)
    	else:
    		image = cv2.imread(img_name)

    	image = cv2.resize(image,(self.dim,self.dim))
    	category = self.category[idx]
    	if self.transform:
    		image = self.transform(image)
    	return image,category

transform = transforms.Compose([
								transforms.ToTensor(),
								transforms.ToPILImage(),
                                transforms.ColorJitter(),
								transforms.RandomRotation(60, resample=False, expand=False, center=None),
								transforms.RandomHorizontalFlip(),
								transforms.RandomVerticalFlip(),
								transforms.ToTensor()
								])


transform2 = transforms.Compose([transforms.ToTensor()])

trainset = PanIITDataset(csv_file='./solution.csv',root_dir='./training',transform = transform)
submissionset = PanIITDataset(csv_file='./sample.csv',root_dir='./testing',transform = transform2)

train_dataset, test_dataset = torch.utils.data.random_split(trainset, [000, 5000])
trainloader = DataLoader(dataset=train_dataset, batch_size=64,shuffle=True, num_workers=4)
testloader = DataLoader(dataset=test_dataset, batch_size=64,shuffle=False, num_workers=4)
submissionloader = DataLoader(dataset=submissionset, batch_size=2,shuffle=False, num_workers=0)

model = models.resnet18(True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 6)
)
model = torch.load('./resnet.pt')
model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.0001,momentum=0.9)

# for epoch in range(0):
# 	ave_loss=[]
# 	for i,(inputs,labels)in enumerate(trainloader):
# 		inputs,labels = inputs.cuda(),labels.cuda()
# 		optimizer.zero_grad()
# 		output=model(inputs)
# 		loss=criterion(output,labels-1)
# 		loss.backward()
# 		optimizer.step()
# 		ave_loss.append(loss.item())
# 		if i%50==49:  
# 			torch.save(model,'./resnet.pt')
# 	print('epoch {0} , loss {1}'.format(epoch+1,np.mean(ave_loss)))


# correct=0
# total=0
# with torch.no_grad():
# 	for i,(inputs,labels)in enumerate(testloader):
# 		inputs,labels = inputs.cuda(),labels.cuda()
# 		outputs=model(inputs)
# 		_,predicted=torch.max(outputs.data,1)
# 		total+=labels.size(0)
# 		correct+=(predicted==labels-1).sum().item()

# print('Accuracy of the network on the  test images: %0.6f %%' % (
#     100 * correct / total))


print('id,category')
i=1
with torch.no_grad():
    for data in submissionloader:
        inputs,labels= data
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs =model(inputs)
        _,predicted=torch.max(outputs.data,1)
        print(str(i)+','+str(int(predicted[0]+1)))
        print(str(i+1)+','+str(int(predicted[1]+1)))
        i=i+2