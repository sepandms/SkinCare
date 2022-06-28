
import torch
import torch.nn as nn
device = torch.device('cpu')

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