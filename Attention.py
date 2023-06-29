import torch
import torch.nn as nn
import numpy as np

# Define the attention-based fusion module
class AttentionFusion(nn.Module):
        def __init__(self, dim):

            """
            In the constructor we instantiate two weight parameters W1 and W2 as well as a kernel q.
            """
            super(AttentionFusion,self).__init__()
            size=dim
            weight1 = torch.randn(())
            weight2=torch.randn(())
            self.W1 = nn.Parameter(weight1, requires_grad=True)
            self.W2 = nn.Parameter(weight2, requires_grad=True)
            kernel=torch.randn(size)
            self.q = nn.Parameter(kernel, requires_grad=True) #2,2 is the shape of the inputs #

            torch.nn.init.uniform_(self.W1, a=0.0, b=1.0)
            torch.nn.init.uniform_(self.W2)
            torch.nn.init.uniform_(self.q)

        def forward(self, rgb, msr):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            kernel=(self.q)
            kernel = kernel.view(-1)
            flatrgb=rgb.view(-1)
            flatmsr=msr.view(-1)
            output1 = torch.matmul(flatrgb, kernel) #Perform the dot product
            output2 = torch.matmul(flatmsr, kernel)

            output1=torch.exp(output1)/torch.exp(output1+output2) #Softmax
            output2=torch.exp(output2)/torch.exp(output1+output2)

            output1= output1* self.W1 #Scalar product
            final1=output1 * rgb
            stream1=(final1)
            

            output2= output1* self.W1
            final2=output2 * rgb
            stream2=(final2)

            return stream1+stream2


