# purpose: change the names of all the faces files to begin with their original
#   filenames, allowing for all the files of all the faces to be combined
#   into one directory so that the metadata script can be run in order

from natsort import natsorted, ns
import os
import re

# modified to change the names back to being just numbers
#remove_lower = lambda text: re.sub('[a-z]', '', text)

text = open("filenames1.txt")
textArr = text.read().splitlines() # put file into array


path = "faces_1"
myFiles = os.listdir("faces_1")
# sort naturally since the order of listdir is arbitrary
myFiles = natsorted(myFiles)

textCount = 0
# for all the files in the faces directory...
for count, filename in enumerate(myFiles):

	# until you get a match between the number of video from the array
	# 	and the number of .jpg, add one to the number video in array
	while str(int(filename.split('_')[0])) != str(textCount):
		textCount += 1

	# get the name of the video from the array
	#   and clean the .jpg from the name
	vidName = textArr[textCount].split('.')[0]

	# used to change the names back to just being numbers
	#newName = filename.split('.')[0]
	#newName = remove_lower(newName)
	#newName = newName + ".jpg"

	# append the video name to the filename
	newName = vidName + filename
	print(newName)

	# reset the filename
	# os.rename(src, dst)
	os.rename('faces_1/' + filename, 'faces_1/' + newName)

	#if filename.split('_')[0] != str(textCount):
		#textCount += 1

text.close()
