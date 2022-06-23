import torch
import torch.nn as nn

class Net8_a(nn.Module):
    def _init_(self,drop_out=0.33):
        super()._init_()
        self.conv1 = nn.Conv2d( in_channels=3 , out_channels=128, stride = 2 , kernel_size=(5, 5))
        self.pool = nn.MaxPool2d( kernel_size = (2,2), stride = 2, padding = 0 )
        self.conv2 = nn.Conv2d( in_channels=128, out_channels=256, stride = 2 , kernel_size=(4, 4))
        self.conv3 = nn.Conv2d( in_channels=256, out_channels=512, stride = 2 , kernel_size=(3, 3))
        self.conv4 = nn.Conv2d( in_channels=512, out_channels=512, stride = 1 , kernel_size=(3, 3))
        self.fc1   = nn.Linear(in_features= 1024 , out_features = 256)
        self.fc2   = nn.Linear(in_features= 256 , out_features = 7)
        self.Act   = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(drop_out)
        self.fcDrop = nn.Dropout(drop_out)
        self.norm = nn.BatchNorm2d(3)

    def forward_train(self, x):

        x = self.norm(x)
        out = self.Act(self.conv1(x))
        out = self.pool(out)
        out = self.dropout(out)

        out = self.Act(self.conv2(out))
        out = self.pool(out)
        out = self.dropout(out)

        out = self.Act(self.conv3(out))
        out = self.Act(self.conv4(out))
        out = self.pool(out)
        out = self.dropout(out)

        out = torch.flatten(out, 1) 
        out = self.Act(self.fc1(out))
        out = self.fcDrop(out)
        out = self.fc2(out)
        return out

    def forward_noDrop(self, x):
        x = self.norm(x)
        out = self.Act(self.conv1(x))
        out = self.pool(out)
        out = self.dropout(out)
        out = self.Act(self.conv2(out))
        out = self.pool(out)
        out = self.Act(self.conv3(out))
        out = self.Act(self.conv4(out))
        out = self.pool(out)
        out = torch.flatten(out, 1) 
        out = self.Act(self.fc1(out))
        out = self.fc2(out)
        return out