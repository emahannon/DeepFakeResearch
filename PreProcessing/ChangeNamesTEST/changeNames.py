# purpose: change the names of all the faces files to begin with their original
#   filenames, allowing for all the files of all the faces to be combined
#   into one directory so that the metadata script can be run in order

from natsort import natsorted, ns
import os

text = open("filenames0.txt")
textArr = text.read().splitlines() # put file into array


path = "faces_0"
myFiles = os.listdir("faces_0")
# sort naturally since the order of listdir is arbitrary
myFiles = natsorted(myFiles)

textCount = 0
# for all the files in the faces directory...
for count, filename in enumerate(myFiles):

	# get the name of the video from the array
	#   and clean the .mp4 from the name
	vidName = textArr[textCount].split('.')[0]

	# append the video name to the filename
	newName = vidName + filename
	print(newName)

	# reset the filename
	# os.rename(src, dst)
	os.rename('faces_0/' + filename, 'faces_0/' + newName)

	if filename.split('_')[0] != str(textCount):
		textCount += 1

text.close()
