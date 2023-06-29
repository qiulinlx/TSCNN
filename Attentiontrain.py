import torch
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import Attention as at
import regtrans as rt
import numpy as np
import RetinexAlgorithm as ret

os.chdir('/home/user/folder/datasets/')

# Define the path to the .pth model file
model_path1 = "./modelrgb.pth"
model_path2 = "./modelmsr.pth"

# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Create the ImageFolder dataset
dataset = ImageFolder(root='./rgbdataset/train/', transform=transform)
testset = ImageFolder(root='./msrdataset/val/', transform=rt.mbvtransform)

#Some parameters
num_epochs = 10
class_labels = dataset.classes
batch_size = 16
test_size = batch_size
num_epochs = 10

variance_list=[15, 80, 30]
variance=300

# Create a DataLoader for batch processing
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=test_size, shuffle=True)


# Load the CNN models
rgbmodel = torch.load(model_path1)
msrmodel = torch.load(model_path2)

# Optional: If the model was saved on a GPU, you can load it onto a specific device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rgbmodel = rgbmodel.to(device)
msrmodel = msrmodel.to(device)

rgbmodel.eval()
msrmodel.eval()

#Initialise attention module and functions to train it
criterion = nn.BCEWithLogitsLoss()
attention = at.AttentionFusion([batch_size, 2])
optimizer = optim.SGD(attention.parameters(), lr=0.001, momentum=0.9)

'''Training Attention module'''
for epoch in range(num_epochs):
    attention.train()
    for images, labels in dataloader:
        # Convert labels to one-hot encoding
        onelabels = F.one_hot(labels, num_classes=len(class_labels))
        onelabels = onelabels.float()

        msrimg=[] #Convert images to msr format
        rt.msrconversion(images, msrimg)
        msrimg=np.array(msrimg)
        msrimg=torch.tensor(msrimg)
        
        msrimg=msrimg.permute(0, 3, 1, 2)
        msrimg = msrimg.to(torch.float32)

        #Feed images into the two CNNs
        rgbfeature = rgbmodel(images)
        msrfeature = msrmodel(msrimg)

        #Feed output of the two CNNs into the attention module
        output = attention(rgbfeature, msrfeature)
        
        #Forward Pass
        loss = criterion((output), onelabels)
        optimizer.zero_grad()

        #Perform backward pass
        loss.backward()
        optimizer.step()
        print("I am training. Pls hold")

torch.save(attention, 'attentionmodule.pth') #Save model for testing 

#attention= torch.load('attentionmodule.pth')

attention.eval()
with torch.no_grad():

    for images, labels in testloader:

        #Convert labels to one-hot encoding
        onelabels = F.one_hot(labels, num_classes=len(class_labels))
        onelabels = onelabels.float()

        msrimg=[]
        rt.msrconversion(images, msrimg)
        msrimg=np.array(msrimg)
        msrimg=torch.tensor(msrimg)
        
        msrimg=msrimg.permute(0, 3, 1, 2)
        msrimg = msrimg.to(torch.float32)

        rgbfeature = rgbmodel(images)
        msrfeature = msrmodel(msrimg)
        pred_outputs = attention(rgbfeature, msrfeature) 

        predicted_labels = torch.round(torch.sigmoid(pred_outputs)) #sigmoid produces probabilities that are rounded to 0 or 1

        # Generate TP, TN, FP, FN for confusion matrix
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(10):
            data = predicted_labels[i]
            adata = onelabels[i]
            if data[0] == adata[0] == 0 and data[1] == adata[1] == 1:
                TN += 1
            if data[0] == adata[0] == 1 and data[1] == adata[1] == 0:
                TP += 1
            if data[0] == adata[1] == 1:
                FP += 1
            if data[1] == adata[0] == 1:
                FN += 1
        con_matrix = [[TP, FP], [FN, TN]]
        print("Confusion Matrix:", con_matrix)

        print("Precision Score:", TP/(FP+TP))

        # Visualize results of one image
        pred = predicted_labels[0]
        one=onelabels[0]
        if one[0] == 0 and one[1] == 1:  # Define the label
            label = "spoof"
        else:
            label = "real"

        # Convert the tensor to a NumPy array
        image_array = images[0].numpy()
        image_array = np.transpose(image_array, (1, 2, 0))

        plt.imshow(image_array)
        plt.title(label)
        plt.text(0, 0, ("Pr this is a real image:",
                 torch.sigmoid(pred_outputs[0][0])), color='g')
        plt.axis('off')  # Turn off axes
        plt.show()
