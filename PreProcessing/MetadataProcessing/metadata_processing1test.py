import json5
import shutil
import glob

# this will load the json file as a dictionary
with open('1/metadata.json') as jsonFile:
	file = json5.load(jsonFile)


# if you want to sort in the same order as the files were processed,
#   put into a list and use the .sort function
sortedKeys = sorted(file.keys())

print(file.keys())

# the first print should give a sorted list
print("Sorted keys:")
print(sortedKeys)
print("\n")

# this should print all the key, value pairs in sorted order
keyNum = 0
for key in sortedKeys:
	#print(key, " ", file[key])
	#print(file[key]['is_fake'])
	# myKey = 'dfdc_train_part_0/' + key
	myKey = glob.glob('faces_1/' + str(keyNum) + '_*.jpg')
	#print(myKey)
	# assuming that 0 is real and 1 is fake for is_fake
	if (file[key]['is_fake'] == 1):
		for name in myKey:
			shutil.copy(name, 'fakes_1')
			#print("fake detected")
	if (file[key]['is_fake'] == 0):
		for name in myKey:
			shutil.copy(name, 'reals_1')
			#print("real detected")
	keyNum += 1;

print("\n")
print(file)
