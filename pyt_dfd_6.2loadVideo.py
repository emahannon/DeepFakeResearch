from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import glob
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ffmpeg
import av
import torchvision.models as models

# THIS VERSION WILL LOAD VIDEOS DIECTLY WITH TORCHVISION
# IT SHOULD ALSO BE NOTED THAT ANYWHERE IMAGES ARE MENTIONED, VIDEOS ARE NOW BEING USED

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
print("Pythorch version: ")
print(torchvision.__version__)
print("Loading full videos with resnet 3D")
print("Loads 0-20 training set")

cl_vids_lst = glob.glob('video_reals_part_0/*mp4')
cl_vids_lst += glob.glob('video_reals_part_1/*mp4')
cl_vids_lst += glob.glob('video_reals_part_2/*mp4')
cl_vids_lst += glob.glob('video_reals_part_3/*mp4')
cl_vids_lst += glob.glob('video_reals_part_4/*mp4')
cl_vids_lst += glob.glob('video_reals_part_5/*mp4')
cl_vids_lst += glob.glob('video_reals_part_6/*mp4')
cl_vids_lst += glob.glob('video_reals_part_7/*mp4')
cl_vids_lst += glob.glob('video_reals_part_8/*mp4')
cl_vids_lst += glob.glob('video_reals_part_9/*mp4')
cl_vids_lst += glob.glob('video_reals_part_10/*mp4')
cl_vids_lst += glob.glob('video_reals_part_11/*mp4')
cl_vids_lst += glob.glob('video_reals_part_12/*mp4')
cl_vids_lst += glob.glob('video_reals_part_13/*mp4')
cl_vids_lst += glob.glob('video_reals_part_14/*mp4')
cl_vids_lst += glob.glob('video_reals_part_15/*mp4')
cl_vids_lst += glob.glob('video_reals_part_16/*mp4')
cl_vids_lst += glob.glob('video_reals_part_17/*mp4')
cl_vids_lst += glob.glob('video_reals_part_18/*mp4')
cl_vids_lst += glob.glob('video_reals_part_19/*mp4')
cl_vids_lst += glob.glob('video_reals_part_20/*mp4')

cl_gt_label = [1] * len(cl_vids_lst)

print("length of the clean and fake images list:")
print(len(cl_vids_lst))
print(len(cl_vids_lst), flush=True)
print("done ", flush=True)

# these are the fakes
#fk_imgs_lst = glob.glob('fakes_part_2/*jpg')
#fk_imgs_lst = glob.glob('./all_frames_face_samebb/fs/*jpg')
#fk_imgs_lst += glob.glob('./all_frames_face_samebb/nt/*jpg')
#fk_imgs_lst += glob.glob('./all_frames_face_samebb/f2f/*jpg')
#fk_imgs_lst += glob.glob('./all_frames_face_samebb/df/*jpg')
fk_imgs_lst = glob.glob('video_fakes_part_0/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_1/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_2/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_3/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_4/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_5/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_6/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_7/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_8/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_9/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_10/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_11/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_12/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_13/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_14/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_15/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_16/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_17/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_18/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_19/*mp4')
fk_imgs_lst += glob.glob('video_fakes_part_20/*mp4')

fk_gt_label = [0] * len(fk_imgs_lst)

print(len(fk_imgs_lst))
print(len(fk_imgs_lst) + len(cl_vids_lst))

tr_imgs = []
tr_lbl = []
tst_imgs = []
tst_lbl = []

# finish setting up testing images
tr_imgs += cl_vids_lst
tr_imgs += fk_imgs_lst

tr_lbl += cl_gt_label
tr_lbl += fk_gt_label


# set up training images
# DOUBLE CHECK THAT THE PATHS TO THE TRAINING VIDEOS ARE ACCURATE
tst_imgs_rl = glob.glob('../test/video_reals/*mp4')
tst_label_rl = [1] * len(tst_imgs_rl)
tst_imgs_fk = glob.glob('../test/video_fakes/*mp4')
tst_label_fk = [0] * len(tst_imgs_fk)

tst_imgs += tst_imgs_rl
tst_imgs += tst_imgs_fk

tst_lbl += tst_label_rl
tst_lbl += tst_label_fk

# 000_0000
# usrs_frame
# last image: 999_0334


print(len(tr_lbl))

# data loader class
# this dataloader needs to be modified to return the right form to go into resnet
# although this tensor from the video stuff may be okay because it was incompatable
# with the other form
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

		#image = io.imread(self.imgs_list[idx])
		# read_video returns: Tensor[T, H, W, C]
		image = torchvision.io.read_video(self.imgs_list[idx])
		image = image[0]

		print(image.shape[0])
		if(image.shape[0] == 0):
			idx = idx + 1
			image = torchvision.io.read_video(self.imgs_list[idx])
			image = image[0]

		skip = 2

		# this will go 16 frames at a time
		numFrames = 16
		print(image.shape[0]-skip*numFrames)

		# get a random number here
		# THE PROBLEM IS WITH THE LINE BELOW ON TEST vvv
		startFrame = np.random.randint(0, image.shape[0]-skip*numFrames)
		image = image[startFrame:startFrame+skip*numFrames:skip]
		lbl = self.label_list[idx]

		# get the height and width of the video
		# (height, width)

		# FORM EXPECTED BY transforms:
		#(B, C, H, W) shape, where B is a number of images in the batch
		# for us, B is T, which is the number of video frames
		image = image.permute(0,3,1,2)
		## lines were used for the commented out transforms below
		dim = image.shape
		#print(dim, flush=True)

		# get 50% of the height
		##dim1 = int(.2 * dim[2])

		#print(dim1, flush=True)
		#print(dim[3], flush=True)


		# creating a composition of a transformation:
		# take random SQUARE crop of image tensor
		# size = some fized percentage of smaller side ex 90% of height
		# AND resize the image tensor to a fixed size
		t = transforms.Compose([
			#transforms.RandomCrop((dim1, dim[3])),
			#transforms.Resize((dim1, dim[3]))
			transforms.Resize((int(256), int(256))),
			transforms.RandomCrop((int(224), int(224)))
			])

		# apply the transformations
		# t(image).size
		image = t(image)
		image = image.permute(0,2,3,1)
		#print(image.shape, flush=True)
		#print("done ", flush=True)

		sample = {'image': image, 'lbl': lbl}

		# return the output tensor and label
		return sample

print("initializing data")
# initialize data set
train_dataset = df_data(imgs_list=tr_imgs, label_list=tr_lbl)
test_dataset = df_data(imgs_list=tst_imgs, label_list=tst_lbl)

BATCH_SIZE = 1

# need to create a new dataloader for the testing/validation that is equal parts

# initialize data loader
# num workers == number of cpus
# loads data w parallel processing
# GUESSING I NEED TO REPLACE THE LINES BELOW WITH THE VIDEO SPECIFIC DATA
# 	LOADING FUNCTION FROM torchvision.io read_video OR VideoReader function?
print("runnning with 0 workers")
dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
# train_dataset and test_dataset should contain lists of the video paths to the dataset
# does this need to be a for loop?
#dataloader = []
#testloader = []

#for video in train_dataset:
	#print(video)
	#dataloader.append(torchvision.io.read_video(video))
	#testloader.append(torchvision.io.read_video(video))

#dataloader = torchvision.io.read_video(train_dataset)
#testloader = torchvision.io.read_video(train_dataset)
# not sure if I have to make modifications elsewhere or if this will "just work"

print("got to here")

for i in range(len(train_dataset)):
	#sample = train_dataset[i]
	#print(i, sample['image'].shape, sample['lbl'])
	break

# not sure if this is right
for i in range(len(test_dataset)):
	#sample = test_dataset[i]
	#print(i, sample['image'].shape, sample['lbl'])
	break

# class to create neural net
"""
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
"""
#net = ConvNet() # create net
net = models.video.r3d_18(pretrained=True)
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

print("beginning training")
lossForChart = []
# 1 loop through for is 1 epoch
# too many epochs will overfit
for epoch in range(1):  # loop over the dataset multiple times

	running_loss = 0.0
	correct = 0

	# training
	for i, data in enumerate(dataloader):
	#for i, data in enumerate(testloader):
		# extract img + label
		# get the inputs; data is a list of [inputs, labels]
		inputs = data['image'].to(device)

		labels = data['lbl'].to(device)
		#inputs = inputs.permute(0,3,1,2).float().to(device) # put img on gpu
		inputs = inputs.permute(0,4,2,3,1).float().to(device) # put img on gpu
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
				lossForChart.append(loss.item())

print('Finished Training')



# now test
# iterate over the test set and make a prediction
# use torch no grad here ??
# compute metricse
test_loss = 0
test_correct = 0

# how do you save what you have once you run?
# for example: torch.save(model, 'aerialmodel.pth')

lossForChartTest = []

net.eval()
with torch.no_grad():
	for i, data in enumerate(testloader):
		inputs = data['image'].to(device)
		labels = data['lbl'].to(device)
		#inputs = inputs.permute(0,3,1,2).float().to(device) # put img on gpu
		inputs = inputs.permute(0,4,2,3,1).float().to(device) # put img on gpu

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
		lossForChartTest.append(loss.item())

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
print(test_loss)
print(accuracy)
print(test_correct)

print('\nTest set: Avg loss:{:.6f} Accuracy on test:{:.6f} Correct: {:.6f}'.format(
	float(test_loss), float(accuracy), float(test_correct)))

print('\nTraining loss plot:\n')
plt.plot(lossForChart)
plt.show()

print('\nTesting loss plot:\n')
plt.plot(lossForChartTest)
plt.show()
