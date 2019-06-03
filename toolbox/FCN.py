import torch
import torch.nn as nn
import torch.nn.functional as F
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(512, 512 )
        #self.fc2 = nn.Linear(512, 512)
        self.Dropout1 = nn.Dropout(p=0.5)
        self.Dropout2 = nn.Dropout(p=0.5)
        self.output = nn.Sigmoid()
        self.classifier = nn.Linear(512, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))        
        #x = self.Dropout1(x)
        #x = F.relu(self.fc2(x))
        #x = self.Dropout2(x)
        #x = self.output(self.fc3(x))
        #x = self.fc4(x)
        x = self.classifier(x)
        return x