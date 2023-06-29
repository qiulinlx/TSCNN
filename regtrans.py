# Define the transformations to be applied to the images
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import RetinexAlgorithm as ret
import torch

variance_list=[15, 80, 30]
variance=300
msrimg=[]

mbvtransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
'''Resizes and normalises the images in the dataset so they can be fed into the Mobilenet model'''

rsnettransform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])

def msrconversion(images, msrimg):
        '''convert into msr features in each batch
        input: RGB images (tensor) from dataloader
               Empty list to store transformed tensor
        Returns: MSR images (list) same name as empty list '''
        to_pil = transforms.ToPILImage()
        msrimages = [to_pil(tensor) for tensor in images]
        for image in msrimages:
            msrimages=ret.MSR(image,variance_list)
            msrimg.append(msrimages)
