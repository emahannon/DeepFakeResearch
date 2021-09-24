import json5
import shutil
import glob


with open('test_all' + '/metadata.json') as jsonFile:
	trainFile = json5.load(jsonFile)

sortedKeys = sorted(trainFile.keys())

keyNum = 0
for key in sortedKeys:
	#print(key, " ", file[key])
	#print(file[key]['label'])
	myKey = 'test_all/' + str(key)
	#myKey = glob.glob(myKey)
	#print(myKey)
	if (trainFile[key]['is_fake'] == 1):
		shutil.copy(myKey, 'video_fakes')
	if (trainFile[key]['is_fake'] == 0):
		shutil.copy(myKey, 'video_reals')
	keyNum += 1;
