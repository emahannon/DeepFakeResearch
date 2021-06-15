import cv2

import glob
# change to the file path on the line below
# going to try to start w test set then train then validation
lst = glob.glob("test/*.mp4")

lst.sort()

import numpy as np

max1 = 0
t = 0
for i in lst:
		print(t)
		vidcap = cv2.VideoCapture(i)
		success,image = vidcap.read()
		count = 0
		while success:
			success,image = vidcap.read()
			count += 1
			if(success):
					cv2.imwrite("testFrames/original/%s_%s.jpg" % ('{:03}'.format(t),'{:03}'.format(count)), image)     # save frame as JPEG file
		t = t+1
# change to line 23 on the file path
