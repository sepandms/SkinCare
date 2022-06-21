
import torch
import torch.nn as nn

class CNN_Nets:
  class Net1(nn.Module):
      def __init__(self):
          super().__init__()
          self.conv1 = nn.Conv2d( in_channels=3 , out_channels=32 , kernel_size=(3, 3))
          self.conv2 = nn.Conv2d( in_channels=32, out_channels=32 , kernel_size=(3, 3))
          self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
          self.conv3 = nn.Conv2d( in_channels=32, out_channels=64 , kernel_size=(3, 3))
          self.conv4 = nn.Conv2d( in_channels=64, out_channels=64 , kernel_size=(3, 3))
          self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
          self.fc1   = nn.Linear(in_features= 202752 , out_features = 512)
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
          out = self.self.fc2(out)
          # out = nn.Softmax(out)
          return out

  class Net2(nn.Module):
      def __init__(self):
          super().__init__()
          self.conv1 = nn.Conv2d( in_channels=3 , out_channels=32 , kernel_size=(3, 3))
          self.pool1 = nn.MaxPool2d( kernel_size = (2,2), stride = None, padding = 0 )
          self.conv3 = nn.Conv2d( in_channels=32, out_channels=64 , kernel_size=(3, 3))
          self.pool2 = nn.MaxPool2d( kernel_size = (2,2), stride = None, padding = 0 )
          self.fc1   = nn.Linear(in_features= 7488 , out_features = 512)
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

  class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d( in_channels=3 , out_channels=32, stride = 2 , kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.conv2 = nn.Conv2d( in_channels=32, out_channels=64, stride = 2 , kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.fc1   = nn.Linear(in_features= 12288 , out_features = 256)
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

  class Net4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d( in_channels=3 , out_channels=32, stride = 2 , kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.conv2 = nn.Conv2d( in_channels=32, out_channels=64, stride = 2 , kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.fc1   = nn.Linear(in_features= 12288 , out_features = 256)
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

  class Net5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d( in_channels=3 , out_channels=16 , stride=2, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.fc1   = nn.Linear(in_features= 56304 , out_features = 7)
        self.Act   = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        out = self.Act(self.conv1(x))
        out = self.Act(self.pool1(out))
        out = self.dropout(out)
        out = torch.flatten(out, 1) 
        out = self.fc1(out)
        return out

  class Net6(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d( in_channels=3 , out_channels=16, stride = 2 , kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d( kernel_size = (4,4), stride = 2, padding = 0 )
        self.fc1   = nn.Linear(in_features= 55488 , out_features = 256)
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

  class Net7(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d( in_channels=3 , out_channels=64, stride = 2 , kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.conv2 = nn.Conv2d( in_channels=64, out_channels=128, stride = 2 , kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.conv3 = nn.Conv2d( in_channels=128, out_channels=64, stride = 2 , kernel_size=(3, 3))
        self.pool3 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.fc1   = nn.Linear(in_features= 384 , out_features = 128)
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

  class Net9(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d( in_channels=3 , out_channels=64, stride = 2 , kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.conv2 = nn.Conv2d( in_channels=64, out_channels=128, stride = 2 , kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.conv3 = nn.Conv2d( in_channels=128, out_channels=64, stride = 2 , kernel_size=(3, 3))
        self.pool3 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.fc1   = nn.Linear(in_features= 384 , out_features = 128)
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
  class Net8(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d( in_channels=3 , out_channels=64, stride = 2 , kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.conv2 = nn.Conv2d( in_channels=64, out_channels=128, stride = 2 , kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.conv3 = nn.Conv2d( in_channels=128, out_channels=64, stride = 2 , kernel_size=(3, 3))
        self.pool3 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.fc1   = nn.Linear(in_features= 384 , out_features = 128)
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

  class Net10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d( in_channels=3 , out_channels=64, stride = 1 , kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.conv2 = nn.Conv2d( in_channels=64, out_channels=128, stride = 1 , kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.conv3 = nn.Conv2d( in_channels=128, out_channels=64, stride = 1 , kernel_size=(3, 3))
        self.pool3 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.conv4 = nn.Conv2d( in_channels=64, out_channels=32, stride = 1 , kernel_size=(3, 3))
        self.pool4 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.fc1   = nn.Linear(in_features= 4480 , out_features = 7)
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

  class Net11(nn.Module):
    def __init__(self):
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
        self.fc1 = nn.Linear(1080, 140)
        #self.fc2 = nn.Linear(700, 70)
        self.fc3 = nn.Linear(140, 7)
        self.Act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.Act(self.conv1(x))
        x = self.Act(self.conv2(x))
        x = self.pool(x)
        x = self.dropConv(x)
        x = self.Act(self.conv3(x))
        x = self.Act(self.conv4(x))
        x = self.pool(x)
        x = self.dropConv(x)
        x = self.Act(self.conv5(x))
        x = self.pool(x)
        x = self.dropConv(x)
        x = torch.flatten(x, 1)
        x = self.Act(self.fc1(x))
        x = self.dropFc(x)
        x = self.fc3(x)
        return x
    def forward_noDrop(self, x):
        x = self.Act(self.conv1(x))
        x = self.Act(self.conv2(x))
        x = self.pool(x)
        x = self.Act(self.conv3(x))
        x = self.Act(self.conv4(x))
        x = self.pool(x)
        x = self.Act(self.conv5(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.Act(self.fc1(x))
        x = self.fc3(x)
        return x
        
  class Net8_a(nn.Module):
    def __init__(self,drop_out):
        super().__init__()
        self.conv1 = nn.Conv2d( in_channels=3 , out_channels=64, stride = 2 , kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.conv2 = nn.Conv2d( in_channels=64, out_channels=128, stride = 2 , kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.conv3 = nn.Conv2d( in_channels=128, out_channels=64, stride = 2 , kernel_size=(3, 3))
        self.pool3 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.fc1   = nn.Linear(in_features= 384 , out_features = 128)
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
  
  class Net8_b(nn.Module):
    def __init__(self,drop_out):
        super().__init__()
        self.conv1 = nn.Conv2d( in_channels=3 , out_channels=64, stride = 2 , kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.conv2 = nn.Conv2d( in_channels=64, out_channels=128, stride = 2 , kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.conv3 = nn.Conv2d( in_channels=128, out_channels=256, stride = 2 , kernel_size=(3, 3))
        self.pool3 = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = 0 )
        self.fc1   = nn.Linear(in_features= 1536 , out_features = 128)
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

class Model_Training_with_loader:

    def __init__(self, Net, Drop, LR, batch_size , Momentum, epochs, weight_decay, loss_func, opt_func,w_sampler, trainDataset, validDataset, X_test,Y_test, print_epochs,hyper_params):    
        
        self.model = Net(Drop).to(device)
        if opt_func is torch.optim.Adam:
            self.opt = opt_func(self.model.parameters(), lr=LR, weight_decay=weight_decay)
        else:
            self.opt = opt_func(self.model.parameters(), lr=LR,momentum=Momentum, weight_decay=weight_decay)

        self.loss_func = loss_func()
        self.epochs = epochs
        self.print_epochs = print_epochs
        self.batch_size = batch_size
        self.Epochs_Train_loss = []
        self.Epochs_Train_Acc = []
        self.Epochs_Val_loss = []
        self.Epochs_Val_Acc = []
        self.Epochs_test_loss = []
        self.Epochs_test_Acc = []
        self.hyper_params = hyper_params
        self.Y_test = Y_test
        self.X_test = X_test
        self.train_loader = DataLoader(dataset = trainDataset , sampler = w_sampler, batch_size = self.batch_size, num_workers=4)
        self.valid_loader = DataLoader(dataset = validDataset , shuffle=True, batch_size = self.batch_size, num_workers=2)
        self.test_loader = DataLoader(dataset = testDataset , shuffle=True, batch_size = self.batch_size, num_workers=2)

    def train(self):
        
        model = self.model
        loss_fn = self.loss_func
        opt = self.opt 
        batch_size = self.batch_size

        for epoch in range(1, self.epochs+1 ):
            start_time=time.time()
            steps_train_loss = []
            steps_train_Acc = []
            steps_val_loss = []
            steps_val_Acc = []
            steps_test_loss = []
            steps_test_Acc = []
            torch.cuda.empty_cache()
            for batch, (X, Y) in enumerate(self.train_loader):
                X = X.to(device)
                Y = Y.to(device)
                opt.zero_grad()
                model.train()
                y_pred = model.forward(X)
                loss = loss_fn(y_pred, Y)
                loss.backward()
                opt.step()
                y_pred = y_pred.argmax(axis=1)
                nr_of_corrects = (y_pred == Y).sum().item()
                step_acc = nr_of_corrects / batch_size
                steps_train_Acc.append(step_acc)
                steps_train_loss.append(loss.item())

                # if (i+1) % 200 == 0:    # print every 2000 mini-batches
                #     print('[Epoch: {}, Nr. Batch: {}]  , Train-Steps-loss: {:.1f} , running_acc: {:.1%}'.format(epoch , i+1 , train_steps_loss , batch_nr_correct / train_nr_total))
                #     self.train_steps_acc = []
                #     train_steps_loss = 0

              #validation loss calculation
            for batch, (X, Y) in enumerate(self.valid_loader):
                X = X.to(device)
                Y = Y.to(device)
                model.eval()
                Y_pred = model(X)
                loss_ = loss_fn(Y_pred, Y)
                epoch_loss = loss_.item()
                Y_pred = Y_pred.argmax(axis=1)
                nr_correct = (Y_pred == Y).sum().item()
                step_acc = nr_correct / batch_size
                steps_val_Acc.append(step_acc)
                steps_val_loss.append(epoch_loss)
              
            #Test Set Performance
            for batch, (X, Y) in enumerate(self.test_loader):
                X = X.to(device)
                Y = Y.to(device)
                model.eval()
                Y_pred = model(X)
                loss_ = loss_fn(Y_pred, Y)
                epoch_loss = loss_.item()
                Y_pred = Y_pred.argmax(axis=1)
                nr_correct = (Y_pred == Y).sum().item()
                step_acc = nr_correct / batch_size
                steps_test_Acc.append(step_acc)
                steps_test_loss.append(epoch_loss)

            # Epoch Performance Metrics
            train_epoch_loss = mean(steps_train_loss)
            train_epoch_Acc = mean(steps_train_Acc)
            self.Epochs_Train_loss.append(train_epoch_loss)
            self.Epochs_Train_Acc.append(train_epoch_Acc)   
            val_epoch_loss = mean(steps_val_loss)
            val_epoch_Acc = mean(steps_val_Acc)
            self.Epochs_Val_loss.append(val_epoch_loss)
            self.Epochs_Val_Acc.append(val_epoch_Acc)
            test_epoch_loss = mean(steps_test_loss)
            test_epoch_Acc = mean(steps_test_Acc)           
            self.Epochs_test_loss.append(test_epoch_loss)
            self.Epochs_test_Acc.append(test_epoch_Acc)
            End_time = time.time() 
            if self.print_epochs:
                print(f'[Epoch: {epoch}]  , Train_loss: {train_epoch_loss:.1f} , Train_Acc: {train_epoch_Acc:.1%}, Val_loss: {val_epoch_loss:.1f} , Val_Acc: {val_epoch_Acc:.1%}, Test_Acc: {test_epoch_Acc:.1%}  , run time: {np.round(End_time - start_time, 2)}')
        # print('Finished Training')