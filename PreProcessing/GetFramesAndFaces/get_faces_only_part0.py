import numpy as np
import cv2
import glob
import os
from mtcnn import MTCNN

# if this doesn't work i might lose my mind
# extract only the frames from the training videos

# initialize
lst = glob.glob("dfdc_train_part_0/*.mp4")

max1 = 0
t = 0

lst.sort() # is this even necessary? leave in as a precaution
print("libraries imported and vars initialized")

# extract the video from image and save as a jpg
for i in lst:
	print(t)
	vidcap = cv2.VideoCapture(i) # open the video
	success,image = vidcap.read() # creates an array of images from the video
	# ^ returns a tuple return_value, image
	count = 0

	# this should loop through the entire video frame by frame
	while success:
		success,image = vidcap.read()
		count += 1
		# if a frame has been successfully extracted, then save it
		# do we have to resize the images before saving them, or will we resize in the algo??
		if(success):
			cv2.imwrite("testFrames/original/%s_%s.jpg" % ('{:03}'.format(t),'{:03}'.format(count)), image)     # save frame as JPEG file
	t = t+1
