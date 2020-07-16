import torch_geometric.nn as Gnn
import torch
import torch.nn as nn
from layers import CustomGCNConv
class GCN_Classifier(torch.nn.Module):
    def __init__(self,In,Out,Hidden):
        super(GCN_Classifier,self).__init__()
        self.Conv1 =  CustomGCNConv(In,Hidden)
        #self.Conv2 = CustomGCNConv(Hidden,Hidden)
        self.Final_Linear = torch.nn.Linear(Hidden,Out)
    def forward(self,X,Edge_Index):
        X = self.Conv1(X,Edge_Index)
        X = torch.nn.functional.log_softmax(X,dim=1)
        return X
