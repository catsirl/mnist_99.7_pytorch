# -*- coding: utf-8 -*-

# 导入库

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
import torchvision.transforms as transforms
import torchvision

# 导入Mnist

transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))])
trainset = torchvision.datasets.MNIST('mnist', train=True, 
                    download=True,
                    transform=transform)
testset = torchvision.datasets.MNIST('mnist', train = False,
                    download=True,
                    transform=transform)
train_loader = torch.utils.data.DataLoader(trainset,
                      batch_size = 100,
                      shuffle = True,
                      num_workers=0)
test_loader = torch.utils.data.DataLoader(testset,
                      batch_size = 100,
                      shuffle = False,
                      num_workers=0)

# 模型定义

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.features = nn.Sequential(
          nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1), 
          nn.ReLU(inplace=True), 
          nn.BatchNorm2d(32),
          nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1), 
          nn.ReLU(inplace=True), 
          nn.BatchNorm2d(32),
          nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1), 
          nn.MaxPool2d(2, 2),
          nn.ReLU(inplace=True), 
          nn.BatchNorm2d(32),
          nn.Dropout(p=0.4),

          nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1), 
          nn.ReLU(inplace=True),
          nn.BatchNorm2d(64),
          nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1), 
          nn.ReLU(inplace=True),
          nn.BatchNorm2d(64),
          nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1), 
          nn.MaxPool2d(2, 2),
          nn.ReLU(inplace=True),
          nn.BatchNorm2d(64),
          nn.Dropout(p=0.4)
        )

        self.classifier = nn.Sequential(
          nn.Linear(7 * 7 * 64, 128),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.4),
          nn.Linear(128, 10)
          )

    def forward(self, x):
      
        # Apply the feature extractor in the input
        x = self.features(x)
        
        # Squeeze the three spatial dimensions in one
        x = x.view(-1, 7 * 7 * 64)
        
        # Classify the images
        x = self.classifier(x)
        return x


batch_size = 100
epochs = 30

num_networks = 10
model_list = []
for i in range(num_networks):
    model = Net()
    model_list.append(model)

# 数据增广
RandAffine = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2))
transform = transforms.Compose([
                RandAffine,                
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, )),
                ])

for n_network in range(num_networks):

  model = model_list[n_network]
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(),lr = 0.001)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)

	for epoch in range(epochs):

		trainset = torchvision.datasets.MNIST('mnist', train=True, 
		                download=True,
		                transform=transform)
		train_loader = torch.utils.data.DataLoader(trainset,
		                  batch_size = 100,
		                  shuffle = True,
		                  num_workers=10)
    	model.train()
		for i, data in enumerate(train_loader, 0):
			train, labels = data
			train, labels = train.to(device), labels.to(device)

			optimizer.zero_grad()
			outputs = model(train)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

	correct = 0
	total = 0
	transform1 = transforms.Compose([transforms.ToTensor(),
	            transforms.Normalize((0.5, ), (0.5, ))])
	trainset = torchvision.datasets.MNIST('mnist', train=True, 
	            download=True,
	            transform=transform1)
	train_loader = torch.utils.data.DataLoader(trainset,
	              batch_size = 100,
	              shuffle = True,
	              num_workers=10)
	model.eval()
	with torch.no_grad():
	for i, data in enumerate(train_loader, 0):
		train, labels = data
		train, labels = train.to(device), labels.to(device)
		outputs = model(train)
		predicted = torch.max(outputs.data, 1)[1]
		total += len(labels)
		correct += (predicted == labels).sum()

	accuracy = 100 * correct / float(total)
	print('Network: {} Accuracy: {}%'.format(n_network+1, accuracy))


# 在测试集上测试
transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))])
testset = torchvision.datasets.MNIST('mnist', train = False,
                    download=True,
                    transform=transform)
test_loader = torch.utils.data.DataLoader(testset,
                      batch_size = 10000,
                      shuffle = False,
                      num_workers=10)
correct = 0
total = 0
output_all = torch.empty(10000, 10, num_networks)

for n_network in range(num_networks):
	model = model_list[n_network]
	model.eval()
	with torch.no_grad():
		for i, data in enumerate(test_loader, 0):
			test, labels = data
			test, labels = test.to(device), labels.to(device)
			outputs = model(test)
			output_all[:,:,n_network] = outputs.data

a, _ = torch.max(output_all, 2)
_, predicted_final = torch.max(a, 1)
for i, data in enumerate(test_loader, 0):
	test, labels = data
	test, labels = test.to(device), labels
	total += len(labels)
	correct += (predicted_final == labels).sum()
accuracy = 100 * correct / float(total)
print('Test Accuracy: {}%'.format(accuracy))


# 显示识别错误的图片
for i, data in enumerate(test_loader, 0):
	test, labels = data
	test, labels = test.to(device), labels
	error_index = ((predicted_final == labels)==False).nonzero()

from matplotlib import pyplot as plt
figure = plt.figure()
num_of_images = test[error_index].shape[0]
for index in range(1, num_of_images+1):
	plt.subplot(4, 7, index)
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
	plt.imshow(testset.data[error_index[index-1][0]], cmap='gray_r')
	plt.title('{} {}'.format(testset.targets[error_index[index-1][0]].item(), predicted_final[error_index[index-1]].item()))
	plt.suptitle('28 incorrectly identified images', y=1.05)


# 保存模型
from google.colab import drive
drive.mount('/content/drive')
PATH = './drive/My Drive/'
for i in range(0,10):
	torch.save(model_list[i], PATH+'model'+str(i)+'.pkl')

# 导入模型
from google.colab import drive
drive.mount('/content/drive')
PATH = './drive/My Drive/'
model_list_import = []
for i in range(0,10):
	model = torch.load(PATH+'model'+str(i)+'.pkl')
	model_list_import.append(model)