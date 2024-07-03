# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:04:35 2024

@author: MAT-Admin
"""
import os
import cv2
import util
from util import get_car, read_license_plate, write_csv

current_working_directory = os.getcwd()

file_path = current_working_directory + '\\last_Img.jpg'
frame = cv2.imread(file_path)
#cv2.imshow('Frame',frame)
# print(frame)

license_plate_crop = frame
license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
_, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

# read license plate number
license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
print(license_plate_text)
#cv2.imshow('Frame',license_plate_crop_thresh)