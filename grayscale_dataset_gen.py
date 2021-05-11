import os
import pathlib as pl 
import cv2
import sys
import numpy as np

if __name__ == '__main__':
	
	folder = sys.argv[1]
	files = os.listdir(folder)
	for file in files:
		if ".png" in file:
			rgb = cv2.imread(os.path.join(folder,file))
			gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
			gray3chanel = np.dstack((gray,gray,gray))
			# rewrite image
			cv2.imwrite(os.path.join(folder,file),gray3chanel)
		