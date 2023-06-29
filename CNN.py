import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import regtrans as rt
from torchvision import transforms

os.chdir('/home/user/folder/datasets/')
model_name='modelmsr.pth'
dataset = ImageFolder(root='./msrdataset/train/', transform=rt.mbvtransform) #Load test and dataset into function
testset = ImageFolder(root='./msrdataset/val/', transform=rt.mbvtransform)

batch_size=16
test_size=10

# Get the class labels
class_labels = dataset.classes

# Create a DataLoader for batch processing
dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True)

#Define mobilenet Model
model = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
#model = models.resnet152(weights="ResNeXt50_32X4D_Weights.IMAGENET1K_V2")

model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

#Freeze ALL layers in Mobilenet except the last one
for name, param in model.named_parameters(): 
    if "classifier" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

#Define some parameters to train the CNN
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs=10

for epoch in range(num_epochs):
    model.train()

    # Iterate over the data
    for images, labels in dataloader:

        # Convert labels to one-hot encoding
        onelabels = F.one_hot(labels, num_classes=len(class_labels))
        onelabels=onelabels.float()
        
        #Perform forward pass
        output=model(images)
        loss = criterion((output), onelabels)
        optimizer.zero_grad()

        #Perform backward pass
        loss.backward()
        optimizer.step()


torch.save(model, model_name)

testloader=DataLoader(testset, batch_size=test_size, shuffle=True)

model.eval()
with torch.no_grad():

    for images, labels in testloader:


        # Convert labels to one-hot encoding
        onelabels = F.one_hot(labels, num_classes=len(class_labels))
        onelabels=onelabels.float()
        #print(onelabels)

        pred_outputs = model(images)
        predicted_labels= torch.round(torch.sigmoid(pred_outputs)) #sigmoid produces probabilities that are rounded to 0 or 1

        #Generate TP, TN, FP, FN for confusion matrix
        TP=0
        TN=0
        FP=0
        FN=0
        for i in range(10):
            data= predicted_labels[i]
            adata=onelabels[i]
            if data[0]==adata[0]==0 and data[1]==adata[1]==1:
                TN+=1
            if data[0]==adata[0]==1 and data[1]==adata[1]==0:
                TP+=1
            if data[0]==adata[1]==1:
                FP+=1
            if data[1]==adata[0]==1:
                FN+=1
        con_matrix= [[TP,FP],[FN,TN]]
        print("Confusion Matrix:", con_matrix)

        print("Precision Score:",TP/(FP+TP))


        #Visualize results of one image 
        one=onelabels[0]

        if one[0] == 0 and one[1] == 1: #Define the label
            label = "spoof"
        else:
            label = "real"
        # Convert the tensor to a NumPy array
        image_array = images[0].numpy()
        image_array = np.transpose(image_array, (1, 2, 0))

        plt.imshow(image_array)
        plt.title(label)
        plt.text(0,0, ("Pr this is a real image:", torch.sigmoid(pred_outputs[0][0])) , color='g')
        plt.axis('off')  # Turn off axes
        plt.show()
