import torch
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops,degree

class CustomGCNConv(gnn.MessagePassing):
    '''Implementing Custom GCN Conv'''
    def __init__(self,In,Out):
        super(CustomGCNConv,self).__init__(aggr="add")
        self.Linear = torch.nn.Linear(In,Out)
    def forward(self,X,Edge_Index):
        Edge_Index,_ = add_self_loops(Edge_Index,num_nodes=X.size(0))
        X = self.Linear(X)
        Row,Col = Edge_Index
        Degree = degree(Col,X.size(0),dtype=X.dtype)
        Inv_Sqrt = Degree.pow(-0.5)
        Norm = Inv_Sqrt[Row]*Inv_Sqrt[Col]
        return self.propagate(Edge_Index,x=X,norm=Norm)
    def message(self,x_j,norm):
        Computed_Message = norm.view(-1,1)*x_j
        return Computed_Message

class EdgeConv(gnn.MessagePassing):
    '''Implementing Edge Convolution from Scratch'''
    def __init__(self,In,Out):
        super(EdgeConv,self).__init__(aggr='max')
        self.Linear_In  = torch.nn.Linear(2*In,Out)
        self.Linear_Out = torch.nn.Linear(Out,Out)
    def forward(self,X,Edge_Index):
        return self.propagate(Edge_Index,x=X)
    def message(self,X_i,X_j):
        Computed = [X_i,X_j - X_i]
        Computed = torch.cat(Computed,dim=1)
        Computed = torch.nn.Functional.relu(self.Linear_In(Computed))
        Computed = self.Linear_Out(Computed)
        return Computed