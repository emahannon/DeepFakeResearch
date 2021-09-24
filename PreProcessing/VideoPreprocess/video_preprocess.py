import json5
import shutil
import glob



def processor(numPart):
	with open('dfdc_train_part_' + str(numPart) + '/metadata.json') as jsonFile:
		trainFile = json5.load(jsonFile)

	sortedKeys = sorted(trainFile.keys())

	keyNum = 0
	for key in sortedKeys:
		#print(key, " ", file[key])
		#print(file[key]['label'])
		myKey = 'dfdc_train_part_' + str(numPart) + '/' + str(key)
		#myKey = glob.glob(myKey)
		#print(myKey)
		if (trainFile[key]['label'] == "FAKE"):
			shutil.copy(myKey, 'video_fakes_part_' + str(numPart))
		if (trainFile[key]['label'] == "REAL"):
			shutil.copy(myKey, 'video_reals_part_' + str(numPart))
		keyNum += 1;


for key in range (0, 10):
	processor(key)
