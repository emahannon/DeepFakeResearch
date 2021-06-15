import numpy as np
import cv2
import glob
import os
from mtcnn import MTCNN
#from facenet_pytorch import MTCNN

print("got to here1")
# we want to do a 2 in 1 with 30 frames at a time
# initialize
lst = glob.glob("faceExtractTest/*.mp4")
detector = MTCNN()
max1 = 0
tFrames = 0
tFaces = 0
frames = [] # this shoud just make an empty array but idk if this is the
# ^^ correct data type to store it ??
faces = []
# height = []
# width = []
x = []
y = []

lst.sort() # is this even necessary? leave in as a precaution
print("got to here top")

# is t being used for the same things in both codes?
# figure out how to read in the videos in a loop + figure out how to loop
#   through the videos just by 60 frames at a time
# firat convert to images, then go through the images and find faces 60 at a time
for i in lst:
	print("got to here")
	print(tFrames)
	vidcap = cv2.VideoCapture(i) # open the video
	success, image = vidcap.read() # creates an array of images from the video
	# ^ returns a tuple return_value, image
	count = 0
	# this should loop through the entire video frame by frame
	while success:
		success, image = vidcap.read()
		count +=1
		# for every 60 frames perform detection
		# do a min maz on total coords from all the faces
		# you chould have a box that surrounds ALL the faces
		# resize to 128x128  + make sure you don't mess up the aspect ratio
		# DO NOT DO THIS BLINDLY add out from the center
		# because we want the camera to be fixed, so the face is what is moving
		# we do not want the camera to be following the face
		# cuboid crop
		# make sure you save as jpg
		if(success):
			# by this point whatever is stored in image is one frame

			# we just need to figure out how to perform face detection on more than
			#   one image at a time
			# the variable image is the thing you need to pass
			# but first we need to create the batch of the correct number of frames
			# OR THE LAST IMAGE OF THE VIDEO IS REACHED
			length = len(frames)
			if (length < 60):
				#img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
				frames.append(image)

			# OR IF THE LAST IMAGE IN THE VIDEO IS REACHED
			if (length == 60):
				# loop through the video and for 60 frames at a time perform detection
				#faces = detector(frames)
				for j in frames:
					print(tFaces)
					#img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
					result = detector.detect_faces(j)
					if(len(result)==0):
						tFaces=tFaces+1
						continue
					bb = result[0]['box']
					#bb1 = [max(bb[0],0),max(bb[1],0),max(bb[2],0),max(bb[3],0)]

					#img1 = img[bb1[1]:bb1[1]+bb1[3],bb1[0]:bb1[0]+bb1[2]]
					# resizes the images to 128 x 128
					#img2 = cv2.resize(img1, (128,128), interpolation = cv2.INTER_AREA)

					tFaces = tFaces+1
					#faces.append(img2)
					# height.append(img2.shape[0])
					# width.append(img2.shape[1])
					x.append(bb[0])
					y.append(bb[1])

				# take out the faces and size them correctly
				# 	what format will it give the faces in
				# then save them as images

				# get the dimensions,, should return as tuples
				# .shape from opencv returns a tuple of the rows, cols, and
				# 	color channels as a tuple
				# get the max of each tuple using max()
				maxX = max(x)
				maxY = max(y)
				minX = min(x)
				minY = min(y)

				frameWrite = 0

				# now draw the box using these values
				for k in frames:
					#img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
					# i have notes for how i figured this line out but basically
					# we are trying to crop so that
					# we have (x1,y1) as the top-left and (x2,y2) as the bottom-right
					# where img[y1:y2, x1:x2]
					frameWrite +=1
					bb = k[minY:maxY, minX:maxX]
					# not sure if bb1 is necessary ?
					#bb1 = [max(bb[0],0),max(bb[1],0),max(bb[2],0),max(bb[3],0)]
					#img1 = k[bb1[1]:bb1[1]+bb1[3],bb1[0]:bb1[0]+bb1[2]]
					img2 = cv2.resize(bb, (128,128), interpolation = cv2.INTER_AREA)
					print("writing frames... %d / 60" % (frameWrite))
					cv2.imwrite("faceExtractTest/%s.jpg" % (k[-11:-4]), cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

				# reset frames so there is no overflow
				# + we start with a new batch of 60 frames
				frames = []
				# all these images should be teh same size in the beginning
				# so does .shape start from the left and increase as it goes on ?
				# it doesnt make sense bc we will still have extras in the bottom
				# bc this is based on the faces cropping,
				# so do we need to take the faces cropping and enlarge from there?
				# how do we know which directions to enlarge in?


				# after all the faces have been extracted, find the largest
				# dimensions and set all to those dimensions
				# so, we want to perform this on the original images

				# how to resize so you center on the actual correct dimensions ??



	tFrames = tFrames+1
