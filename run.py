import torch
from model import GCN_Classifier
In_Dim,Out_Dim,Hidden_Dim = 10,20,6
X = torch.randn(3,In_Dim)
Edge_Index = torch.tensor([[1,0],[0,2]])
Classifier = GCN_Classifier(In_Dim,Out_Dim,Hidden_Dim)
print(Classifier(X,Edge_Index))