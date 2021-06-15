import numpy as np
import cv2

import os
from mtcnn import MTCNN

detector = MTCNN()

import glob
lst = glob.glob("all_frames_data/original/*.jpg")

lst.sort()

t = 0
# does this loop go through the entire video ??
for i in lst:
		print(t)
		img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
		result = detector.detect_faces(img)
		if(len(result)==0):
				t=t+1
				continue
		bb = result[0]['box']
		bb1 = [max(bb[0],0),max(bb[1],0),max(bb[2],0),max(bb[3],0)]

		img1 = img[bb1[1]:bb1[1]+bb1[3],bb1[0]:bb1[0]+bb1[2]]
		# resizes the images to 128 x 128
		img2 = cv2.resize(img1, (128,128), interpolation = cv2.INTER_AREA)

		cv2.imwrite("all_frames_face_data/original/%s.jpg" % (i[-11:-4]), cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
		t=t+1
