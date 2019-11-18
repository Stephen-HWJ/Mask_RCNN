import cv2
import os
labelNames = [x for x in os.listdir('.') if ".png" in x] #new
for name in labelNames:
	img = cv2.imread(name)
	img_new = cv2.resize(img, (400, 300))
	cv2.imwrite('../'+name, img_new)