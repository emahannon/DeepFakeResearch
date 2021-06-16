from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# modify to test on the test set:
# 	new dataload er and send test images

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

# glob lists all files of a certain type
# so, we will have a list of all the file paths for our images
# these are the real images
# 60007
cl_imgs_lst = glob.glob('reals_part_2/*jpg')
# reals pt 0: 21410
cl_imgs_lst += glob.glob('reals_part_0/*jpg')
cl_imgs_lst += glob.glob('reals_part_1/*jpg')
cl_imgs_lst += glob.glob('reals_part_3/*jpg')
cl_imgs_lst += glob.glob('reals_part_4/*jpg')
cl_imgs_lst += glob.glob('reals_part_5/*jpg')
# list the number of images
cl_gt_label = [1] * len(cl_imgs_lst)

# total should be 81417

print("length of the clean and fake images list:")
print(len(cl_imgs_lst))

# these are the fakes
# 397557
fk_imgs_lst = glob.glob('fakes_part_2/*jpg')
# fakes pt 0: 271815
fk_imgs_lst += glob.glob('fakes_part_0/*jpg')
fk_imgs_lst += glob.glob('fakes_part_1/*jpg')
fk_imgs_lst += glob.glob('fakes_part_3/*jpg')
fk_imgs_lst += glob.glob('fakes_part_4/*jpg')
fk_imgs_lst += glob.glob('fakes_part_5/*jpg')
#fk_imgs_lst = glob.glob('./all_frames_face_samebb/fs/*jpg')
#fk_imgs_lst += glob.glob('./all_frames_face_samebb/nt/*jpg')
#fk_imgs_lst += glob.glob('./all_frames_face_samebb/f2f/*jpg')
#fk_imgs_lst += glob.glob('./all_frames_face_samebb/df/*jpg')
fk_gt_label = [0] * len(fk_imgs_lst)

# fakes and reals of the test set
tst_fk_imgs_lst = glob.glob('../test/fakes_0/*jpg')
tst_fk_imgs_lst += glob.glob('../test/fakes_1/*jpg')
tst_fk_imgs_lst += glob.glob('../test/fakes_2/*jpg')
tst_fk_imgs_lst += glob.glob('../test/fakes_3/*jpg')
tst_fk_imgs_lst += glob.glob('../test/fakes_4/*jpg')

tst_fk_gt_label = [0] * len(tst_fk_imgs_lst)


tst_cl_imgs_lst = glob.glob('../test/reals_0/*jpg')
tst_cl_imgs_lst += glob.glob('../test/reals_1/*jpg')
tst_cl_imgs_lst += glob.glob('../test/reals_2/*jpg')
tst_cl_imgs_lst += glob.glob('../test/reals_3/*jpg')
tst_cl_imgs_lst += glob.glob('../test/reals_4/*jpg')

tst_cl_gt_label = [1] * len(tst_cl_imgs_lst)

# total shoud be 669372

print("fakes:")
print(len(fk_imgs_lst))
print("reals:")
print(len(cl_imgs_lst))
print()
print("test fakes:")
print(len(tst_fk_imgs_lst))
print("test reals:")
print(len(tst_cl_imgs_lst))


tr_imgs = []
tr_lbl = []
tst_imgs = []
tst_lbl = []


# 000_0000
# usrs_frame
# last image: 999_0334

# split into train and test sets
# clean images
c = 0
for i in cl_imgs_lst:
	# BE SURE TO CHANGE THE IF STATEMENT DEPENDING ON WHAT SIZE SET YOU USE
	#if(int(i[-12:-9])<750):
	tr_imgs += [i]
	tr_lbl += [cl_gt_label[c]]
	#if (int(c) < 40708):
		#tr_imgs += [i]
		#tr_lbl += [cl_gt_label[c]]
	#else:
		#tst_imgs += [i]
		#tst_lbl += [cl_gt_label[c]]
	#c+=1

# fake images
c = 0
for i in fk_imgs_lst:
	# BE SURE TO CHANGE THE IF STATEMENT DEPENDING ON WHAT SIZE SET YOU USE
	#if(int(i[-12:-9])<750):
	tr_imgs += [i]
	tr_lbl += [fk_gt_label[c]]
	#if (int(c) < 334686):
		#tr_imgs += [i]
		#tr_lbl += [fk_gt_label[c]]
	#else:
		#tst_imgs += [i]
		#tst_lbl += [fk_gt_label[c]]
	#c+=1

# fake and clean test images
for i in tst_cl_imgs_lst:
	tst_imgs += [i]
	tst_lbl += [tst_cl_gt_label]

for i in tst_fk_imgs_lst:
	tst_imgs += [i]
	tst_lbl += [tst_fk_gt_label]


print(len(tr_lbl))

# data loader class
class df_data(Dataset):

	def __init__(self, imgs_list, label_list):

		self.imgs_list = imgs_list
		self.label_list = label_list
		self.on_epoch_end()

	def __len__(self):
		return len(self.imgs_list)

	# on epoch end -> shuffles img
	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.imgs_list))
		#if self.shuffle == True:
		np.random.shuffle(self.indexes)

	# reads img based on img id -> read label + put in dictionary
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		image = io.imread(self.imgs_list[idx])
		lbl = self.label_list[idx]

		sample = {'image': image, 'lbl': lbl}

		return sample

# initialize data set
train_dataset = df_data(imgs_list=tr_imgs, label_list=tr_lbl)
test_dataset = df_data(imgs_list=tst_imgs, label_list=tst_lbl)

BATCH_SIZE = 16

# initialize data loader
# num workers == number of cpus
# loads data w parallel processing
dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

for i in range(len(train_dataset)):
	sample = train_dataset[i]
	print(i, sample['image'].shape, sample['lbl'])
	break

# not sure if this is right
for i in range(len(test_dataset)):
	sample = test_dataset[i]
	print(i, sample['image'].shape, sample['lbl'])
	break

# class to create neural net
class ConvNet(nn.Module):

	# main learning part,, expands the amt of layers or imgs to expand
	# the # of patterns that can be detected
	def __init__(self):
		super(ConvNet, self).__init__()

		# Define various layers here, such as in the tutorial example
		# self.conv1 = nn.Conv2D(...)
		# define conv layers SO in the other functions we can create them
		self.conv1 = nn.Conv2d(in_channels=3,
						out_channels=16,
						kernel_size=3,
						stride=1)
		self.conv2 = nn.Conv2d(in_channels=16,
						out_channels=32,
						kernel_size=3,
						stride=1)
		self.conv3 = nn.Conv2d(in_channels=32,
						out_channels=64,
						kernel_size=3,
						stride=1)
		self.conv4 = nn.Conv2d(in_channels=64,
						out_channels=128,
						kernel_size=3,
						stride=1)

		# define 3 fully connected layers
		# 1st param input dimension - 2nd param output dimension
		# scales down to be able to determine between fake or real
		self.fc1 = nn.Linear(6*6*128, 1024)
		self.fc2 = nn.Linear(1024, 256)
		self.fc3 = nn.Linear(256, 2)


	def forward(self, x):

		# (2,2) param reduces the height and width by taking matrix of 4px
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))

		# relu function introduces nonlinearity
		# reshape -> flatten makes all features into one
		# (this is the function from below)
		x = x.reshape(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

net = ConvNet() # create net
print(net)
net.to(device) # puts net on gpu

# loss function -> weights start as random
# tells us how we perform
# - want to be close to 0
# - works based on probability rather than 1 or 0 and compares the floating
#  	point values to true 1 or 0
criterion = nn.CrossEntropyLoss()
# optim.SGD -> takes loss and tries to reduce loss + update weights
# optimizer/gradient will measure the change in all weights w regard to change
# 	in error, can be thought of as slope of a function, when grad is 0 model
# 	stops learning. when grad higher, model can learn faster
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 1 loop through for is 1 epoch
# too many epochs will overfit
for epoch in range(2):  # loop over the dataset multiple times

	running_loss = 0.0
	correct = 0

	# training
	for i, data in enumerate(dataloader):
	#for i, data in enumerate(testloader):
		# extract img + label
		# get the inputs; data is a list of [inputs, labels]
		inputs = data['image'].to(device)
		labels = data['lbl'].to(device)
		inputs = inputs.permute(0,3,1,2).float().to(device) # put img on gpu
		# zero the parameter gradients (put all grad set to 0)
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		# Total correct predictions
		predicted = torch.max(outputs.data, 1)[1]
		correct += (predicted == labels).sum()
		#print(correct)
		if i % 50 == 0:
				print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
					epoch, i*len(inputs), len(dataloader.dataset), 100.*i / len(dataloader),
									loss.item(), float(correct*100) / float(BATCH_SIZE*(i+1))))

print('Finished Training')



# now test
# iterate over the test set and make a prediction
# use torch no grad here ??
# compute metricse
test_loss = 0
test_correct = 0

# how do you save what you have once you run?
# for example: torch.save(model, 'aerialmodel.pth')

net.eval()
with torch.no_grad():
	for i, data in enumerate(testloader):
		inputs = data['image'].to(device)
		labels = data['lbl'].to(device)
		inputs = inputs.permute(0,3,1,2).float().to(device) # put img on gpu

		# do i need to update inputs
		# do i need to update the gradients?? optimizer.no_grad() ??
		# no grad will prevent interference and leaks into model?
		# predicts for test data by doing a forward pass
		outputs = net(inputs)

		# compute loss (37:34)
		# cross entropy most common loss for classification problems
		# 	loss increases as the predicted probability diverges from label
		# is labels the target variable ??
		loss = F.cross_entropy(outputs, labels, reduction='sum')

		# append loss to overall test loss
		# what does .item() do? where does it come from
		# gets a single tensor
		test_loss += loss.item()

		# get predicted index by selecting max log probability
		# argmax function from numpy -- returns indicies of the
		# 	max element of the array in a particular axis
		# why don't we put input parameter in the example like:
		# torch.argmax(input, dim=None, keepdim=False)
		pred = outputs.argmax(dim=1, keepdim=True)

		# count number of correct predictions in total
		# view_as -> view tensor as asme size as parameter
		# sum returns the sum of all the elements
		# item get a number from a tensor containing a single value
		# eq -> computes element-wise equality
		# 	this will return true if we have a match to the
		# 	test set and false otherwise
		test_correct += pred.eq(labels.view_as(pred)).sum().item()

test_loss /= len(testloader.dataset)
accuracy = 100.0 * test_correct / len(testloader.dataset)

print('\nTest set: Avg loss:%.4f Accuracy on test:%.4 Correct: %.5',
	test_loss, accuracy, test_correct)
