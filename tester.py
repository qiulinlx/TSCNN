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
testset = ImageFolder(root='./Tests/', transform=rt.mbvtransform)

#Some parameters
class_labels = testset.classes
batch_size = 16
test_size = batch_size


testloader = DataLoader(testset, batch_size=test_size, shuffle=True)

# Load all models
model_path1 = "./modelrgb.pth"
model_path2 = "./modelmsr.pth"

rgbmodel = torch.load(model_path1)
msrmodel = torch.load(model_path2)
attention= torch.load('attentionmodule.pth')

# Optional: If the model was saved on a GPU, you can load it onto a specific device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rgbmodel = rgbmodel.to(device)
msrmodel = msrmodel.to(device)
attention=attention.to(device)

#Turn all models onto eval mode

rgbmodel.eval()
msrmodel.eval()
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
