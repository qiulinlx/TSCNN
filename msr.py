#MSR tensor from Image
import sys
import os
import cv2
import RetinexAlgorithm as ret
from PIL import Image
import numpy as np

os.chdir('/home/panda/Techassessment/datasets/')
directory = './spoof/' #This is the directory where the algorithm searches for images to transform
save_directory = '/home/panda/Techassessment/datasets/msr/spoof/' #This is where the algorithm saves the images


variance_list=[15, 80, 30]
variance=300
file_list=os.listdir(directory) 

# Specify the directory where you want to save the images
counter=1

for file_name in file_list:
        # Construct the full file path
        file_path = os.path.join(directory, file_name)
        print("file being processed", file_name)

        # Check if the file is an image (optional)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            
            # Open the image file
            img = cv2.imread(file_path)
            img_msr=ret.MSR(img,variance_list)

            #cv2.imshow('Original', img)
            #cv2.imshow('MSR', img_msr)
            name=f"msrr{counter}.png"
            #save_directory = "'/home/panda/Techassessment/datasets/msr/spoof/"

            counter+=1
            isWritten= cv2.imwrite((name),img_msr)

            if isWritten:
	            print('Image is successfully saved as file.')
            

