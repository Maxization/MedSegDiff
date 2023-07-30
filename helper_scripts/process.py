import os
import sys

path = sys.argv[1]
files = os.listdir(path)

for file in files:
	end_ind = file[:file.find("_output")].rfind("_")
	patient = file[:end_ind]
	
	if not os.path.exists(path + patient):
		os.makedirs(path + patient)
	
	os.rename(path + file, path + patient + "/" + file)

print("Ok!")
		