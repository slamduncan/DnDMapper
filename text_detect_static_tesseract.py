# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
from pytesseract import pytesseract
from support_func import *

CONFIDENCE = 0.0001

# The EAST text requires that your input image dimensions be multiples of 32, so if you choose to adjust your
# --width  and --height  values, make sure they are multiples of 32!

# load the input image and grab the image dimensions
img = cv2.imread("TestImages/DnDMap.jpg")

#gray = get_grayscale(img)
#thresh = thresholding(gray)
#img = thresh

custom_config = r'--oem 3 --psm 11 tessedit_char_whitelist=0123456789'
#d = pytesseract.image_to_string(text_imgs[1], config=custom_config)

h, w, c = img.shape # Only for colour images
#h, w = img.shape
boxes = pytesseract.image_to_boxes(img, config=custom_config)
img_boxed = img
for b in boxes.splitlines():
    b = b.split(' ')
    img_boxed = cv2.rectangle(img_boxed, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

#cv2.imshow('img', img)
cv2.imwrite("TestImages/TestOutput_boxed.jpg", img_boxed)

img_full_data = img
d = pytesseract.image_to_data(img_full_data, output_type=pytesseract.Output.DICT, config=custom_config)
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    cv2.rectangle(img_full_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imwrite("TestImages/TestOutput_full_data.jpg", img_full_data)
cv2.waitKey(0)


#d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
#n_boxes = len(d['text'])
#for i in range(n_boxes):
#    if int(d['conf'][i]) > 60:
#        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#        image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#cv2.imshow('img', img)
#cv2.imwrite("TestOutput.jpg", image)
#cv2.waitKey(0)
