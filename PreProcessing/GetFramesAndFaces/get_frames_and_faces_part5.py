import numpy as np
import cv2
import glob
import os
from mtcnn import MTCNN

# GOAL HERE IS TO CLEAN UP CODE AND TO ORGANIZE THE LOOPS SO THAT WE CAN
# GET THE ENDING FRAMES EVEN IF THERE ARE LESS THAN 60

# we want to do a 2 in 1 with 30 frames at a time
# initialize
lst = glob.glob("dfdc_train_part_5/*.mp4")
detector = MTCNN()
max1 = 0
tFrames = 0
tFaces = 0
frames = [] # this shoud just make an empty array but idk if this is the
faces = []
x = []
y = []
frameWrite = 0

lst.sort() # is this even necessary? leave in as a precaution
print("libraries imported and vars initialized")

# is t being used for the same things in both codes?
# figure out how to read in the videos in a loop + figure out how to loop
#   through the videos just by 60 frames at a time
# firat convert to images, then go through the images and find faces 60 at a time
for i in lst:
	vidcap = cv2.VideoCapture(i) # open the video
	success, image = vidcap.read() # creates an array of images from the video
	# ^ returns a tuple return_value, image
	count = 0
	# this should loop through the entire video frame by frame
	while success:
		success, image = vidcap.read()
		count +=1
		# for every 60 frames perform detection
		if(success):
			# by this point whatever is stored in image is one frame

			# but first we need to create the batch of the correct number of frames
			# OR THE LAST IMAGE OF THE VIDEO IS REACHED
			length = len(frames)
			if (length < 60):
				#img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
				frames.append(image)

		# OR IF THE LAST IMAGE IN THE VIDEO IS REACHED
		if (length == 60 or (success == False and length > 0)):
			# loop through the video and for 60 frames at a time perform detection
			framesCount = 0
			facesInBatch = 0
			for j in frames:
				result = detector.detect_faces(j)
				if(len(result)==0):
					# if no face is detected, remove that face
					tFaces=tFaces+1
					continue
				faces.append(j)
				bb = result[0]['box']

				tFaces = tFaces+1

				x.append(bb[0])
				x.append(bb[0] + bb[2])
				y.append(bb[1])
				y.append(bb[1] + bb[3])
				facesInBatch = facesInBatch + 1


			if(facesInBatch != 0):
				maxX = max(x)
				maxY = max(y)
				minX = min(x)
				minY = min(y)


			# now draw the box using these values
			for k in faces:
				#img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
				# i have notes for how i figured this line out but basically
				# we are trying to crop so that
				# we have (x1,y1) as the top-left and (x2,y2) as the bottom-right
				# where img[y1:y2, x1:x2]
				frameWrite +=1

				height, width, channel = k.shape

				maxX1 = min((maxX + int(maxX*.10)), width)
				maxY1 = min((maxY + int(maxY*.10)), height)
				minX1 = max((minX - int(minX*.10)), 0)
				minY1 = max((minY - int(minY*.10)), 0)

				xDiff = maxX1-minX1
				yDiff = maxY1-minY1

				maxDiff = max(xDiff, yDiff)

				if (xDiff == maxDiff):
					changeBy = ((maxDiff-(yDiff))/2)
					maxY1 = min((maxY1 + changeBy), height)
					minY1 = max((minY1 - changeBy), 0)
				elif (yDiff == maxDiff):
					changeBy = ((maxDiff-(xDiff))/2)
					maxX1 = min((maxX1 + changeBy), width)
					minX1 = max((minX1 - changeBy), 0)

				maxX1 = int(maxX1)
				maxY1 = int(maxY1)
				minX1 = int(minX1)
				minY1 = int(minY1)


				bb = k[minY1:maxY1, minX1:maxX1]

				img2 = cv2.resize(bb, (128,128), interpolation = cv2.INTER_AREA)
				name = str(tFrames) + "_" + str(tFaces) + "_" + str(frameWrite)
				cv2.imwrite("faces_part_5/%s.jpg" % (name), img2)


			# reset frames so there is no overflow
			# + we start with a new batch of 60 frames
			frames = []
			faces = []
			x = []
			y = []
			length = 0


	tFrames = tFrames+1
