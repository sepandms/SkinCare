
import torch
import torch.nn as nn

class CNN_Nets:
    class Net1(nn.Module):
        fc_features = 0
        def __init__(self,drop_out=0.2):
            super().__init__()
            self.conv1 = nn.Conv2d( in_channels=3 , out_channels=32 , kernel_size=(3, 3))
            self.conv2 = nn.Conv2d( in_channels=32, out_channels=32 , kernel_size=(3, 3))
            self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv3 = nn.Conv2d( in_channels=32, out_channels=64 , kernel_size=(3, 3))
            self.conv4 = nn.Conv2d( in_channels=64, out_channels=64 , kernel_size=(3, 3))
            self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.fc1   = nn.Linear(in_features= self.fc_features , out_features = 512)
            self.fc2   = nn.Linear(in_features= 512, out_features = 7)
            self.Act   = nn.ReLU(inplace=True)
        def forward(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv3(out))
            out = self.Act(self.conv4(out))
            out = self.Act(self.pool2(out))
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            # out = nn.Softmax(out)
            return out
        def dimention_set(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv3(out))
            out = self.Act(self.conv4(out))
            out = self.Act(self.pool2(out))
            out = torch.flatten(out, 1) 
            fc_nr = out.shape[1]
            return fc_nr

    class Net2(nn.Module):
        fc_features = 0
        def __init__(self,drop_out=0.2):
            super().__init__()
            self.conv1 = nn.Conv2d( in_channels=3 , out_channels=32 , kernel_size=(3, 3))
            self.pool1 = nn.MaxPool2d( kernel_size = (2,2), stride = None, padding = 0 )
            self.conv3 = nn.Conv2d( in_channels=32, out_channels=64 , kernel_size=(3, 3))
            self.pool2 = nn.MaxPool2d( kernel_size = (2,2), stride = None, padding = 0 )
            self.fc1   = nn.Linear(in_features= self.fc_features , out_features = 512)
            self.fc2   = nn.Linear(in_features= 512, out_features = 7)
            self.Act   = nn.ReLU(inplace=True)
        def forward(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool2(out))
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.Act(self.fc2(out))
            return out
        def dimention_set(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool2(out))
            out = torch.flatten(out, 1) 
            fc_nr = out.shape[1]
            return fc_nr

    class Net3(nn.Module):
        fc_features = 0
        def __init__(self,drop_out=0.2):
            super().__init__()
            self.conv1 = nn.Conv2d( in_channels=3 , out_channels=32, stride = 2 , kernel_size=(3, 3))
            self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv2 = nn.Conv2d( in_channels=32, out_channels=64, stride = 2 , kernel_size=(3, 3))
            self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.fc1   = nn.Linear(in_features= self.fc_features , out_features = 256)
            self.fc2   = nn.Linear(in_features= 256, out_features = 7)
            self.Act   = nn.LeakyReLU(inplace=True)
            self.dropout = nn.Dropout(0.25)
        def forward(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.dropout(out)
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.dropout(out)
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def forward_noDrop(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def dimention_set(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = torch.flatten(out, 1) 
            fc_nr = out.shape[1]
            return fc_nr

    class Net4(nn.Module):
        fc_features = 0
        def __init__(self,drop_out=0.2):
            super().__init__()
            self.conv1 = nn.Conv2d( in_channels=3 , out_channels=32, stride = 2 , kernel_size=(3, 3))
            self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv2 = nn.Conv2d( in_channels=32, out_channels=64, stride = 2 , kernel_size=(3, 3))
            self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.fc1   = nn.Linear(in_features= self.fc_features , out_features = 256)
            self.fc2   = nn.Linear(in_features= 256, out_features = 7)
            self.Act   = nn.LeakyReLU(inplace=True)
            self.dropout = nn.Dropout(0.15)
        def forward(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.dropout(out)
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.dropout(out)
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def forward_noDrop(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def dimention_set(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = torch.flatten(out, 1) 
            fc_nr = out.shape[1]
            return fc_nr

    class Net5(nn.Module):
        fc_features = 0
        def __init__(self,drop_out=0.2):
            super().__init__()
            self.conv1 = nn.Conv2d( in_channels=3 , out_channels=16 , stride=2, kernel_size=(3, 3))
            self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.fc1   = nn.Linear(in_features= self.fc_features , out_features = 7)
            self.Act   = nn.LeakyReLU(inplace=True)
            self.dropout = nn.Dropout(0.2)
        def forward(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.dropout(out)
            out = torch.flatten(out, 1) 
            out = self.fc1(out)
            return out
        def dimention_set(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.dropout(out)
            out = torch.flatten(out, 1) 
            fc_nr = out.shape[1]
            return fc_nr

    class Net6(nn.Module):
        fc_features = 0
        def __init__(self,drop_out=0.2):
            super().__init__()
            self.conv1 = nn.Conv2d( in_channels=3 , out_channels=16, stride = 2 , kernel_size=(3, 3))
            self.pool1 = nn.MaxPool2d( kernel_size = (4,4), stride = 2, padding = 0 )
            self.fc1   = nn.Linear(in_features= self.fc_features , out_features = 256)
            self.fc2   = nn.Linear(in_features= 256 , out_features = 7)
            self.Act   = nn.LeakyReLU(inplace=True)
            self.dropout = nn.Dropout(0.2)
        def forward(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.dropout(out)
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def dimention_set(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.dropout(out)
            out = torch.flatten(out, 1) 
            fc_nr = out.shape[1]
            return fc_nr            

    class Net7(nn.Module):
        fc_features = 0
        def __init__(self,drop_out=0.2):
            super().__init__()
            self.conv1 = nn.Conv2d( in_channels=3 , out_channels=64, stride = 2 , kernel_size=(3, 3))
            self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv2 = nn.Conv2d( in_channels=64, out_channels=128, stride = 2 , kernel_size=(3, 3))
            self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv3 = nn.Conv2d( in_channels=128, out_channels=64, stride = 2 , kernel_size=(3, 3))
            self.pool3 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.fc1   = nn.Linear(in_features= self.fc_features , out_features = 128)
            self.fc2   = nn.Linear(in_features= 128 , out_features = 7)
            self.Act   = nn.LeakyReLU(inplace=True)
            self.dropout = nn.Dropout(0.3)
        def forward(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.dropout(out)
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.dropout(out)
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = self.dropout(out)
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def forward_noDrop(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def dimention_set(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = torch.flatten(out, 1) 
            fc_nr = out.shape[1]
            return fc_nr  

    class Net9(nn.Module):
        fc_features = 0
        def __init__(self,drop_out=0.2):
            super().__init__()
            self.conv1 = nn.Conv2d( in_channels=3 , out_channels=64, stride = 2 , kernel_size=(3, 3))
            self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv2 = nn.Conv2d( in_channels=64, out_channels=128, stride = 2 , kernel_size=(3, 3))
            self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv3 = nn.Conv2d( in_channels=128, out_channels=64, stride = 2 , kernel_size=(3, 3))
            self.pool3 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.fc1   = nn.Linear(in_features= self.fc_features , out_features = 128)
            self.fc2   = nn.Linear(in_features= 128 , out_features = 7)
            self.Act   = nn.LeakyReLU(inplace=True)
            self.dropout = nn.Dropout(0.15)
        def forward(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.dropout(out)
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.dropout(out)
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = self.dropout(out)
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def forward_noDrop(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def dimention_set(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = torch.flatten(out, 1) 
            fc_nr = out.shape[1]
            return fc_nr  

    class Net8(nn.Module):
        fc_features = 0
        def __init__(self,drop_out=0.2):
            super().__init__()
            self.conv1 = nn.Conv2d( in_channels=3 , out_channels=64, stride = 2 , kernel_size=(3, 3))
            self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv2 = nn.Conv2d( in_channels=64, out_channels=128, stride = 2 , kernel_size=(3, 3))
            self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv3 = nn.Conv2d( in_channels=128, out_channels=64, stride = 2 , kernel_size=(3, 3))
            self.pool3 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.fc1   = nn.Linear(in_features= self.fc_features , out_features = 128)
            self.fc2   = nn.Linear(in_features= 128 , out_features = 7)
            self.Act   = nn.LeakyReLU(inplace=True)
            self.dropout = nn.Dropout(0.15)
        def forward(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            # out = self.dropout(out)
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            # out = self.dropout(out)
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            # out = self.dropout(out)
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def forward_noDrop(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def dimention_set(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = torch.flatten(out, 1) 
            fc_nr = out.shape[1]
            return fc_nr  

    class Net10(nn.Module):
        fc_features = 0
        def __init__(self,drop_out=0.2):
            super().__init__()
            self.conv1 = nn.Conv2d( in_channels=3 , out_channels=64, stride = 1 , kernel_size=(3, 3))
            self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv2 = nn.Conv2d( in_channels=64, out_channels=128, stride = 1 , kernel_size=(3, 3))
            self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv3 = nn.Conv2d( in_channels=128, out_channels=64, stride = 1 , kernel_size=(3, 3))
            self.pool3 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv4 = nn.Conv2d( in_channels=64, out_channels=32, stride = 1 , kernel_size=(3, 3))
            self.pool4 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.fc1   = nn.Linear(in_features= self.fc_features , out_features = 7)
            # self.fc2   = nn.Linear(in_features= 128 , out_features = 7)
            self.Act   = nn.LeakyReLU(inplace=True)
            self.dropout = nn.Dropout(0.15)
        def forward(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.dropout(out)
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.dropout(out)
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = self.dropout(out)
            out = self.Act(self.conv4(out))
            out = self.Act(self.pool4(out))
            out = self.dropout(out)
            out = torch.flatten(out, 1) 
            out = self.fc1(out)
            # out = self.fc2(out)
            return out
        def forward_noDrop(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = self.Act(self.conv4(out))
            out = self.Act(self.pool4(out))
            out = torch.flatten(out, 1) 
            out = self.fc1(out)
            return out
        def dimention_set(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = self.Act(self.conv4(out))
            out = self.Act(self.pool4(out))
            out = torch.flatten(out, 1) 
            fc_nr = out.shape[1]
            return fc_nr  


    class Net11(nn.Module):
        fc_features = 0
        def __init__(self,drop_out=0.2):
            super().__init__()
            self.pool = nn.MaxPool2d(2, 2)
            self.dropConv = nn.Dropout2d(0.3)
            self.dropFc = nn.Dropout(0.3)
            self.normalization = nn.BatchNorm2d(3)
            self.conv1 = nn.Conv2d(3, 60, 5, 2)
            self.conv2 = nn.Conv2d(60, 120, 5)
            self.conv3 = nn.Conv2d(120, 150, 4, 2)
            self.conv4 = nn.Conv2d(150, 180, 3)
            self.conv5 = nn.Conv2d(180, 180, 3, 2)
            self.fc1 = nn.Linear(self.fc_features, 140)
            #self.fc2 = nn.Linear(700, 70)
            self.fc3 = nn.Linear(140, 7)
            self.Act = nn.LeakyReLU(inplace=True)

        def forward(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.conv2(out))
            out = self.pool(out)
            out = self.dropConv(out)
            out = self.Act(self.conv3(out))
            out = self.Act(self.conv4(out))
            out = self.pool(out)
            out = self.dropConv(out)
            out = self.Act(self.conv5(out))
            out = self.pool(out)
            out = self.dropConv(out)
            out = torch.flatten(out, 1)
            out = self.Act(self.fc1(out))
            out = self.dropFc(out)
            out = self.fc3(out)
            return out
        def forward_noDrop(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.conv2(out))
            out = self.pool(out)
            out = self.Act(self.conv3(out))
            out = self.Act(self.conv4(out))
            out = self.pool(out)
            out = self.Act(self.conv5(out))
            out = self.pool(out)
            out = torch.flatten(out, 1)
            out = self.Act(self.fc1(out))
            out = self.fc3(out)
            return out
        def dimention_set(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.conv2(out))
            out = self.pool(out)
            out = self.Act(self.conv3(out))
            out = self.Act(self.conv4(out))
            out = self.pool(out)
            out = self.Act(self.conv5(out))
            out = self.pool(out)
            out = torch.flatten(out, 1) 
            fc_nr = out.shape[1]
            return fc_nr  

    class Net8_a(nn.Module):
        fc_features = 0
        def __init__(self,drop_out=0.2):
            super().__init__()
            self.conv1 = nn.Conv2d( in_channels=3 , out_channels=64, stride = 2 , kernel_size=(3, 3))
            self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv2 = nn.Conv2d( in_channels=64, out_channels=128, stride = 2 , kernel_size=(3, 3))
            self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv3 = nn.Conv2d( in_channels=128, out_channels=64, stride = 2 , kernel_size=(3, 3))
            self.pool3 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.fc1   = nn.Linear(in_features= self.fc_features , out_features = 128)
            self.fc2   = nn.Linear(in_features= 128 , out_features = 7)
            self.Act   = nn.LeakyReLU()
            self.dropout = nn.Dropout(drop_out)
        def forward(self, x):
            out = self.Act(self.conv1(x))
            out = self.pool1(self.Act(out))
            out = self.dropout(out)
            out = self.Act(self.conv2(out))
            out = self.pool2(self.Act(out))
            out = self.dropout(out)
            out = self.Act(self.conv3(out))
            out = self.pool3(self.Act(out))
            out = self.dropout(out)
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def forward_noDrop(self, x):
            out = self.Act(self.conv1(x))
            out = self.pool1(self.Act(out))
            out = self.Act(self.conv2(out))
            out = self.pool2(self.Act(out))
            out = self.Act(self.conv3(out))
            out = self.pool3(self.Act(out))
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def dimention_set(self, x):
            out = self.Act(self.conv1(x))
            out = self.pool1(self.Act(out))
            out = self.Act(self.conv2(out))
            out = self.pool2(self.Act(out))
            out = self.Act(self.conv3(out))
            out = self.pool3(self.Act(out))
            out = torch.flatten(out, 1) 
            fc_nr = out.shape[1]
            return fc_nr

    class Net8_a_binary(nn.Module):
        fc_features = 0
        def __init__(self,drop_out=0.2):
            super().__init__()
            self.conv1 = nn.Conv2d( in_channels=3 , out_channels=64, stride = 2 , kernel_size=(3, 3))
            self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv2 = nn.Conv2d( in_channels=64, out_channels=128, stride = 2 , kernel_size=(3, 3))
            self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv3 = nn.Conv2d( in_channels=128, out_channels=256, stride = 2 , kernel_size=(3, 3))
            self.pool3 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.fc1   = nn.Linear(in_features= self.fc_features, out_features = 128)
            self.fc2   = nn.Linear(in_features= 128 , out_features = 2)
            self.Act   = nn.LeakyReLU()
            self.dropout = nn.Dropout(drop_out)
        def forward(self, x):
            out = self.Act(self.conv1(x))
            out = self.pool1(out)
            out = self.dropout(out)
            out = self.Act(self.conv2(out))
            out = self.pool2(out)
            out = self.dropout(out)
            out = self.Act(self.conv3(out))
            out = self.pool3(out)
            out = self.dropout(out)
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def forward_noDrop(self, x):
            out = self.Act(self.conv1(x))
            out = self.pool1(out)
            out = self.Act(self.conv2(out))
            out = self.pool2(out)
            out = self.Act(self.conv3(out))
            out = self.pool3(out)
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def dimention_set(self, x):
            out = self.Act(self.conv1(x))
            out = self.pool1(out)
            out = self.Act(self.conv2(out))
            out = self.pool2(out)
            out = self.Act(self.conv3(out))
            out = self.pool3(out)
            out = torch.flatten(out, 1) 
            fc_nr = out.shape[1]
            return fc_nr

    class Net8_b(nn.Module):
        fc_features = 0
        def __init__(self,drop_out=0.2):
            super().__init__()
            self.conv1 = nn.Conv2d( in_channels=3 , out_channels=64, stride = 2 , kernel_size=(3, 3))
            self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv2 = nn.Conv2d( in_channels=64, out_channels=128, stride = 2 , kernel_size=(3, 3))
            self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv3 = nn.Conv2d( in_channels=128, out_channels=256, stride = 2 , kernel_size=(3, 3))
            self.pool3 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.fc1   = nn.Linear(in_features= self.fc_features , out_features = 128)
            self.fc2   = nn.Linear(in_features= 128 , out_features = 7)
            self.Act   = nn.LeakyReLU(inplace=True)
            self.dropout = nn.Dropout(drop_out)
        def forward(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.dropout(out)
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.dropout(out)
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = self.dropout(out)
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def forward_noDrop(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def dimention_set(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = torch.flatten(out, 1) 
            fc_nr = out.shape[1]
            return fc_nr


    class Net8_b_binary(nn.Module):
        fc_features = 0
        def __init__(self,drop_out=0.2):
            super().__init__()
            self.conv1 = nn.Conv2d( in_channels=3 , out_channels=64, stride = 2 , kernel_size=(3, 3))
            self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv2 = nn.Conv2d( in_channels=64, out_channels=128, stride = 2 , kernel_size=(3, 3))
            self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.conv3 = nn.Conv2d( in_channels=128, out_channels=256, stride = 2 , kernel_size=(3, 3))
            self.pool3 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
            self.fc1   = nn.Linear(in_features= self.fc_features , out_features = 128)
            self.fc2   = nn.Linear(in_features= 128 , out_features = 2)
            self.Act   = nn.LeakyReLU(inplace=True)
            self.dropout = nn.Dropout(drop_out)
        def forward(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.dropout(out)
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.dropout(out)
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = self.dropout(out)
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def forward_noDrop(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = torch.flatten(out, 1) 
            out = self.Act(self.fc1(out))
            out = self.fc2(out)
            return out
        def dimention_set(self, x):
            out = self.Act(self.conv1(x))
            out = self.Act(self.pool1(out))
            out = self.Act(self.conv2(out))
            out = self.Act(self.pool2(out))
            out = self.Act(self.conv3(out))
            out = self.Act(self.pool3(out))
            out = torch.flatten(out, 1) 
            fc_nr = out.shape[1]
            return fc_nr

# MLP


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