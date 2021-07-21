import json5
import shutil
import glob

# this will load the json file as a dictionary
with open('dfdc_train_part_10/metadata.json') as jsonFile:
	file = json5.load(jsonFile)


# if you want to sort in the same order as the files were processed,
#   put into a list and use the .sort function
sortedKeys = sorted(file.keys())

#print(file.keys())

# the first print should give a sorted list
print("process loaded")
#print(sortedKeys)
#print("\n")

# this should print all the key, value pairs in sorted order
keyNum = 0
for key in sortedKeys:
	#print(key, " ", file[key])
	#print(file[key]['label'])
	# myKey = 'dfdc_train_part_0/' + key
	myKey = glob.glob('faces_part_10/' + str(keyNum) + '_*.jpg')
	#print(myKey)
	if (file[key]['label'] == "FAKE"):
		for name in myKey:
			shutil.copy(name, 'fakes_part_10')
	if (file[key]['label'] == "REAL"):
		for name in myKey:
			shutil.copy(name, 'reals_part_10')
	keyNum += 1;

print("\n")
print("process complete")
