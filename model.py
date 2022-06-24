import torch
import torch.nn as nn

class neuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(neuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2= nn.Linear(hidden_size,hidden_size)
        self.l3= nn.Linear(hidden_size,num_classes)
        self.relu=nn.ReLU()

    def forward(self,x):
        out=self.l1(x)
        out=self.relu(out) #içi c'est l'activation aprés chaque Linear "L1"
        out=self.l2(out)
        out=self.relu(out) # activation pour L2
        out=self.l3(out)
        # içi on va pas faire une activation parceque aprés on va appliquer la cross donc ça va activer systématiquement à la fin 
        return out