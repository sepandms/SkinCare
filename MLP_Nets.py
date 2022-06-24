import torch
import torch.nn as nn


class MLP_Nets:
    class Net1(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(in_features= 30 , out_features = 60)
            self.fc2 = nn.Linear(in_features= 60 , out_features = 80)
            self.fc3 = nn.Linear(in_features= 80 , out_features = 7)
            self.Act   = nn.LeakyReLU(inplace=True)
            # self.Act   = nn.Sigmoid()

        def forward(self, x):
            out = self.Act(self.fc1(x))
            out = self.Act(self.fc2(out))
            out = self.fc3(out)
            # print(out)
            # out = F.softmax(out,dim=1)
            return out

    class Net2(nn.Module):

        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(in_features= 30 , out_features = 20)
            self.fc2 = nn.Linear(in_features= 20 , out_features = 14)
            self.fc3 = nn.Linear(in_features= 14 , out_features = 7)
            self.Act   = nn.LeakyReLU(inplace=True)
            self.dropout = nn.Dropout(0.25)
            # self.Act   = nn.Sigmoid()

        def forward(self, x):
            out = self.Act(self.fc1(x))
            out = self.Act(self.fc2(out))
            out= self.dropout(out)
            out = self.fc3(out)
            # print(out)
            # out = F.softmax(out,dim=1)
            return out
            
    class Net3(nn.Module):

        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(in_features= 11 , out_features = 20)
            self.fc2 = nn.Linear(in_features= 20 , out_features = 14)
            self.fc3 = nn.Linear(in_features= 14 , out_features = 7)
            self.Act   = nn.Sigmoid()
            self.dropout = nn.Dropout(0.3)
            # self.Act   = nn.Sigmoid()

        def forward(self, x):
            out = self.Act(self.fc1(x))
            out = self.Act(self.fc2(out))
            out= self.dropout(out)
            out = self.Act(self.fc3(out))
            # print(out)
            # out = F.softmax(out,dim=1)
            return out