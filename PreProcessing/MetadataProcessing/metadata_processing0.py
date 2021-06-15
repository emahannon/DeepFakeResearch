import json5
import shutil
import glob

# this will load the json file as a dictionary
with open('dfdc_train_part_0/metadata.json') as jsonFile:
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
	print(key, " ", file[key])
	#print(file[key]['label'])
	# myKey = 'dfdc_train_part_0/' + key
	myKey = glob.glob('faces_part_2/' + str(keyNum) + '_*.jpg')
	#print(myKey)
	if (file[key]['label'] == "FAKE"):
		for name in myKey:
			shutil.copy(name, 'fakes_part_0')
	if (file[key]['label'] == "REAL"):
		for name in myKey:
			shutil.copy(name, 'reals_part_0')
	keyNum += 1;

print("\n")
print(file)
