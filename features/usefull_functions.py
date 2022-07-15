
import sklearn as sk
import sys
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import pickle
from datetime import datetime

def model_evaluation(Y,Y_pred,title='Confusion Matrix', plot_CM=True):
    CM = sk.metrics.confusion_matrix(Y,Y_pred)
    if plot_CM:
        print('Nr. of Data : \n', CM.sum())
        print('Accuracy of The Model : \n', np.diag(CM).sum()/CM.sum())
        disp = sk.metrics.ConfusionMatrixDisplay(CM)
        disp.plot()
        disp.im_.colorbar.remove()
        font = {'size'   : 14}
        plt.rc('font', **font)
        plt.title(title, fontsize = 16)
        plt.xlabel('Predicted', fontsize = 14)
        plt.ylabel('True', fontsize = 14)
        plt.show()
    FP = CM.sum(axis=0) - np.diag(CM) 
    FN = CM.sum(axis=1) - np.diag(CM)
    TP = np.diag(CM)
    TN = CM.sum() - (FP + FN + TP)
    weights = CM.sum(axis=1) / CM.sum() 
    ACC = np.nan_to_num((TP+TN)/(TP+FP+FN+TN) , nan=0)
    Recall_Sensitivity = np.nan_to_num(TP/(TP+FN) , nan=0)
    Specificity = np.nan_to_num(TN/(TN+FP) , nan=0)
    Precision = np.nan_to_num(TP/(TP+FP) , nan=0)
    f1_score = np.nan_to_num( 2*Precision*Recall_Sensitivity / (Recall_Sensitivity + Precision), nan=0)
    Performance_DF = pd.concat([pd.DataFrame(CM),pd.DataFrame(weights, columns=['weights']),pd.DataFrame(Precision, columns=['Precision']),pd.DataFrame(Recall_Sensitivity,columns=['Recall_Sensitivity'])
        ,pd.DataFrame(Specificity, columns=['Specificity']),pd.DataFrame(f1_score, columns=['f1_score'])], axis=1)
    total_row1 = pd.Series({'Precision':mean(Precision),'Recall_Sensitivity':mean(Recall_Sensitivity),'Specificity':mean(Specificity),'f1_score':mean(f1_score)}, name='Simple Avg.')
    total_row2 = pd.Series({'Precision':sum(weights*Precision),'Recall_Sensitivity':sum(weights*Recall_Sensitivity),'Specificity':sum(weights*Specificity),'f1_score':sum(weights*f1_score)}, name='Weighted Avg.')
    Performance_DF = Performance_DF.append([total_row1,total_row2])
    cols = ['weights','Precision','Recall_Sensitivity','Specificity','f1_score']
    per_details = Performance_DF[cols].style.format({'weights': "{:.1%}",'Precision': "{:.1%}",'Recall_Sensitivity': "{:.1%}",'Specificity': "{:.1%}",'f1_score': "{:.1%}"})
    return per_details


def confusion_matrix(Y,Y_pred):
    CM = sk.metrics.confusion_matrix(Y,Y_pred)
    print('Nr. of Data : \n', CM.sum())
    print('Accuracy of The Model : \n', np.diag(CM).sum()/CM.sum())
    sk.metrics.ConfusionMatrixDisplay(CM).plot()
    FP = CM.sum(axis=0) - np.diag(CM) 
    FN = CM.sum(axis=1) - np.diag(CM)
    TP = np.diag(CM)
    TN = CM.sum() - (FP + FN + TP)
    weights = CM.sum(axis=1) / CM.sum() 
    ACC = np.nan_to_num((TP+TN)/(TP+FP+FN+TN) , nan=0)
    Recall_Sensitivity = np.nan_to_num(TP/(TP+FN) , nan=0)
    Specificity = np.nan_to_num(TN/(TN+FP) , nan=0)
    Precision = np.nan_to_num(TP/(TP+FP) , nan=0)
    f1_score = np.nan_to_num( 2*Precision*Recall_Sensitivity / (Recall_Sensitivity + Precision), nan=0)
    Performance_DF = pd.concat([pd.DataFrame(CM),pd.DataFrame(weights, columns=['weights']),pd.DataFrame(Precision, columns=['Precision']),pd.DataFrame(Recall_Sensitivity,columns=['Recall_Sensitivity'])
        ,pd.DataFrame(Specificity, columns=['Specificity']),pd.DataFrame(f1_score, columns=['f1_score'])], axis=1)
    total_row1 = pd.Series({'Precision':mean(Precision),'Recall_Sensitivity':mean(Recall_Sensitivity),'Specificity':mean(Specificity),'f1_score':mean(f1_score)}, name='Simple Avg.')
    total_row2 = pd.Series({'Precision':sum(weights*Precision),'Recall_Sensitivity':sum(weights*Recall_Sensitivity),'Specificity':sum(weights*Specificity),'f1_score':sum(weights*f1_score)}, name='Weighted Avg.')
    Performance_DF = Performance_DF.append([total_row1,total_row2])
    cols = ['weights','Precision','Recall_Sensitivity','Specificity','f1_score']
    per_details = Performance_DF[cols].style.format({'weights': "{:.1%}",'Precision': "{:.1%}",'Recall_Sensitivity': "{:.1%}",'Specificity': "{:.1%}",'f1_score': "{:.1%}"})
    return per_details




def plot_loss_accuracy(model_):
    epochs_X = [i for i in range(1, model_.epochs+1)]
    fig, axs = plt.subplots(1,2,figsize=(14,4))
    axs[0].plot(epochs_X , model_.Epochs_Train_loss , 'bo-', label='Train loss')
    axs[0].plot(epochs_X , model_.Epochs_Val_loss,'ro-', label='Validation loss')
    axs[0].plot(epochs_X , model_.Epochs_test_loss,'go-', label='Test loss')
    axs[0].set_xlabel("Epochs", fontsize = 12)
    axs[0].set_ylabel("Loss", fontsize = 12)
    axs[0].grid()
    axs[0].legend()
    axs[0].set_title('Train and Validation loss by epochs', fontsize = 14)
    axs[1].plot(epochs_X , model_.Epochs_Train_Acc , 'bo-', label='Train Accuracy')
    axs[1].plot(epochs_X , model_.Epochs_Val_Acc ,'ro-', label='Validation Accuracy')
    axs[1].plot(epochs_X , model_.Epochs_test_Acc ,'go-', label='Test Accuracy')
    axs[1].set_xlabel("Epochs", fontsize = 12)
    axs[1].set_ylabel("Accuracy", fontsize = 12)
    axs[1].grid()
    axs[1].legend()
    axs[1].set_title('Train and Validation Accuracy by epochs', fontsize = 14)
    plt.show()


remap = {0:1 , 1:1, 4:1, 2:0, 3:0, 5:0, 6:0}
def label_to_binary(entry):
    return remap[entry] if entry in remap else entry
label_to_binary = np.vectorize(label_to_binary)


def recall_specificity_precision_fscore(Y,Y_pred, weighted_avg=True):
    CM = sk.metrics.confusion_matrix(Y,Y_pred)
    FP = CM.sum(axis=0) - np.diag(CM) 
    FN = CM.sum(axis=1) - np.diag(CM)
    TP = np.diag(CM)
    TN = CM.sum() - (FP + FN + TP)
    weights = CM.sum(axis=1) / CM.sum() 
    Accuracy = np.nan_to_num((TP+TN)/(TP+FP+FN+TN) , nan=0)
    Recall = np.nan_to_num(TP/(TP+FN) , nan=0)
    Specificity = np.nan_to_num(TN/(TN+FP) , nan=0)
    Precision = np.nan_to_num(TP/(TP+FP) , nan=0)
    if weighted_avg:
        return round(sum(weights*Recall),3), round(sum(weights*Specificity),3), round(sum(weights*Precision),3), round( sum(weights * (2*Recall*Precision/sum(Recall*Precision))) ,3)
    else:
        return round(mean(Recall),3), round(mean(Specificity),3), round(mean(Precision),3) ,  round( mean((2*Recall*Precision/sum(Recall*Precision))) ,3)

def plot_grid_results(model_):
    epochs_X = [i for i in range(1, len( list(model_['train_epoch_loss'])[0]) +1)]
    fig, axs = plt.subplots(1,2,figsize=(14,4))
    axs[0].plot(epochs_X , list(model_.train_epoch_loss)[0] , 'bo-', label='Train loss')
    axs[0].plot(epochs_X , list(model_.valid_epoch_loss)[0],'ro-', label='Validation loss')
    axs[0].plot(epochs_X , list(model_.test_epoch_loss)[0],'go-', label='Test loss')
    axs[0].set_xlabel("Epochs", fontsize = 12)
    axs[0].set_ylabel("Loss", fontsize = 12)
    axs[0].grid()
    axs[0].legend()
    axs[0].set_title('Train and Validation loss by epochs', fontsize = 14)
    axs[1].plot(epochs_X , list(model_.train_epoch_acc)[0] , 'bo-', label='Train Accuracy')
    axs[1].plot(epochs_X , list(model_.valid_epoch_acc)[0] ,'ro-', label='Validation Accuracy')
    axs[1].plot(epochs_X , list(model_.test_epoch_acc)[0] ,'go-', label='Test Accuracy')
    axs[1].set_xlabel("Epochs", fontsize = 12)
    axs[1].set_ylabel("Accuracy", fontsize = 12)
    axs[1].grid()
    axs[1].legend()
    axs[1].set_title('Train and Validation Accuracy by epochs', fontsize = 14)
    plt.show()

def plot_grid_results2(model_):
    epochs_X = [i for i in range(1, len( list(model_['train_epoch_loss'])[0]) +1)]
    fig, axs = plt.subplots(1,3,figsize=(24,4.5))
    axs[0].plot(epochs_X , list(model_.train_epoch_loss)[0] , 'bo-', label='Train loss')
    axs[0].plot(epochs_X , list(model_.valid_epoch_loss)[0],'ro-', label='Validation loss')
    axs[0].plot(epochs_X , list(model_.test_epoch_loss)[0],'go-', label='Test loss')
    axs[0].set_xlabel("Epochs", fontsize = 12)
    axs[0].set_ylabel("Loss", fontsize = 12)
    axs[0].grid()
    axs[0].legend()
    axs[0].set_title('Train and Validation loss by epochs', fontsize = 14)
    axs[1].plot(epochs_X , list(model_.train_epoch_acc)[0] , 'bo-', label='Train Accuracy')
    axs[1].plot(epochs_X , list(model_.valid_epoch_acc)[0] ,'ro-', label='Validation Accuracy')
    axs[1].plot(epochs_X , list(model_.test_epoch_acc)[0] ,'go-', label='Test Accuracy')
    axs[1].set_xlabel("Epochs", fontsize = 12)
    axs[1].set_ylabel("Accuracy", fontsize = 12)
    axs[1].grid()
    axs[1].legend()
    axs[1].set_title('Train and Validation Accuracy by epochs', fontsize = 14)
    

    fpr, tpr, roc_auc = model_['test_fpr'][0], model_['test_tpr'][0], model_['test_roc_auc'][0]
    n_classes = fpr.keys().__len__() -2

    # Plot all ROC curves
    # fig = plt.figure(figsize=(7,5))
    axs[2].plot(fpr["micro"], tpr["micro"],
            label=f'micro ({roc_auc["micro"]:0.2f})' 
            ,color='deeppink', linestyle=':', linewidth=4)

    axs[2].plot(fpr["macro"], tpr["macro"],
            label=f'macro ({roc_auc["macro"]:0.2f})'
            ,color='navy', linestyle=':', linewidth=4)

    for i in range(n_classes):  
        axs[2].plot(fpr[i], tpr[i], linestyle='--', 
                label=f'{i} ({roc_auc[i]:0.2f})')

    axs[2].plot([0, 1], [0, 1], 'k--')
    axs[2].set_xlim([0.0, 1.0])
    axs[2].set_ylim([0.0, 1.05])
    axs[2].set_xlabel('False Positive Rate')
    axs[2].set_ylabel('True Positive Rate')
    axs[2].set_title('ROC: Multi-Class')
    axs[2].legend(loc="lower right")
    plt.show()


# def progressbar(n,tot, prefix="", size=30, out=sys.stdout):
#     count = tot
#     def show(j):
#         x = int(size*j/count)
#         out.write("%s[%s%s] %i/%i\r" % (prefix, u"█"*x, "."*(size-x), j, count))
#         out.flush()        
#     show(0)
#     show(n)
#     out.flush()

def fpr_tpr_score(Y_OneH,Y_pred_prob):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresh = dict()
    n_classes = Y_OneH.shape[1]
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = sk.metrics.roc_curve(Y_OneH[:, i], Y_pred_prob[:, i])
        roc_auc[i] = sk.metrics.auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = sk.metrics.roc_curve(Y_OneH.ravel(), Y_pred_prob.ravel())
    roc_auc["micro"] = sk.metrics.auc(fpr["micro"], tpr["micro"])       
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sk.metrics.auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc

# Process of plotting roc-auc curve belonging to all classes.
def plot_roc_auc_multi(fpr, tpr, roc_auc):

    n_classes = fpr.keys().__len__() -2

    # Plot all ROC curves
    fig = plt.figure(figsize=(7,5))
    plt.plot(fpr["micro"], tpr["micro"],
            label=f'micro ({roc_auc["micro"]:0.2f})' 
            ,color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label=f'macro ({roc_auc["macro"]:0.2f})'
            ,color='navy', linestyle=':', linewidth=4)

    for i in range(n_classes):  
        plt.plot(fpr[i], tpr[i], linestyle='--', 
                label=f'{i} ({roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: Multi-Class')
    plt.legend(loc="lower right")
    plt.show()

def recall_specificity(Y,Y_pred):
    CM = sk.metrics.confusion_matrix(Y,Y_pred)
    FP = CM.sum(axis=0) - np.diag(CM) 
    FN = CM.sum(axis=1) - np.diag(CM)
    TP = np.diag(CM)
    TN = CM.sum() - (FP + FN + TP)
    weights = CM.sum(axis=1) / CM.sum() 
    ACC = np.nan_to_num((TP+TN)/(TP+FP+FN+TN) , nan=0)
    Recall_Sensitivity = np.nan_to_num(TP/(TP+FN) , nan=0)
    Specificity = np.nan_to_num(TN/(TN+FP) , nan=0)
    Precision = np.nan_to_num(TP/(TP+FP) , nan=0)
    f1_score = np.nan_to_num( 2*Precision*Recall_Sensitivity / (Recall_Sensitivity + Precision), nan=0)
    Performance_DF = pd.concat([pd.DataFrame(CM),pd.DataFrame(weights, columns=['weights']),pd.DataFrame(Precision, columns=['Precision']),pd.DataFrame(Recall_Sensitivity,columns=['Recall_Sensitivity'])
        ,pd.DataFrame(Specificity, columns=['Specificity']),pd.DataFrame(f1_score, columns=['f1_score'])], axis=1)
    total_row1 = pd.Series({'Precision':mean(Precision),'Recall_Sensitivity':mean(Recall_Sensitivity),'Specificity':mean(Specificity),'f1_score':mean(f1_score)}, name='Simple Avg.')
    total_row2 = pd.Series({'Precision':sum(weights*Precision),'Recall_Sensitivity':sum(weights*Recall_Sensitivity),'Specificity':sum(weights*Specificity),'f1_score':sum(weights*f1_score)}, name='Weighted Avg.')
    Performance_DF = Performance_DF.append([total_row1,total_row2])
    cols = ['weights','Precision','Recall_Sensitivity','Specificity','f1_score']
    return Performance_DF



class Model_Training_with_loader:

    def __init__(self, Net, Drop, LR, batch_size , Momentum, epochs,patience, weight_decay, loss_func, opt_func,w_sampler, trainDataset, validDataset,testDataset, print_epochs,hyper_params,device):    
        
        self.model = Net(Drop).to(device)
        if opt_func is torch.optim.Adam:
            self.opt = opt_func(self.model.parameters(), lr=LR, weight_decay=weight_decay)
        else:
            self.opt = opt_func(self.model.parameters(), lr=LR,momentum=Momentum, weight_decay=weight_decay)

        self.loss_func = loss_func()
        self.epochs = epochs
        self.patience = patience
        self.print_epochs = print_epochs
        self.batch_size = batch_size
        self.Epochs_Train_loss = []
        self.Epochs_Train_Acc = []
        self.Epochs_Val_loss = []
        self.Epochs_Val_Acc = []
        self.Epochs_test_loss = []
        self.Epochs_test_Acc = []
        self.hyper_params = hyper_params
        self.train_loader = DataLoader(dataset = trainDataset , sampler = w_sampler, batch_size = self.batch_size, num_workers=4)
        self.valid_loader = DataLoader(dataset = validDataset , shuffle=True, batch_size = self.batch_size, num_workers=2)
        self.test_loader = DataLoader(dataset = testDataset , shuffle=True, batch_size = self.batch_size, num_workers=2)
        self.device = device
        print('\n')

    def train(self):
        
        model = self.model
        loss_fn = self.loss_func
        opt = self.opt 
        batch_size = self.batch_size
        min_loss = 100
        iters = 0

        for epoch in range(1, self.epochs+1 ):
            start_time = time.time()
            steps_train_loss = []
            steps_train_Acc = []
            steps_val_loss = []
            steps_val_Acc = []
            steps_test_loss = []
            steps_test_Acc = []
            torch.cuda.empty_cache()
            for batch, (X, Y) in enumerate(self.train_loader):
                X = X.to(self.device)
                Y = Y.to(self.device)
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
                X = X.to(self.device)
                Y = Y.to(self.device)
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
                X = X.to(self.device)
                Y = Y.to(self.device)
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

            if val_epoch_loss < min_loss:
              min_loss = val_epoch_loss
              pickle.dump(model,open('Best_Model','wb'))
              iters = 0
            else:
              iters +=1
     
            if iters > self.patience:
              model = pickle.load(open('Best_Model','rb'))
              print(f'Earlt Stoppo happen at Epoche {epoch} after no improvment of {iters} epochs ')
              break

            if self.print_epochs:
                print(f'[Epoch: {epoch}]  , Train_loss: {train_epoch_loss:.1f} , Train_Acc: {train_epoch_Acc:.1%}, Val_loss: {val_epoch_loss:.1f} , Val_Acc: {val_epoch_Acc:.1%}, Test_Acc: {test_epoch_Acc:.1%}  , run time: {np.round(End_time - start_time, 2)}')
        # print('Finished Training')



def grid_searc_cross_valid_trainer(Model_, grid, cross_valid, kflods ,X_train , Y_train,X_valid,Y_valid,X_test, Y_test,nr_repeat ):

    Param_Details = pd.DataFrame(columns=['hyper_param','valid_recall_weighted','test_recall_weighted'
                                  ,'valid_recall_simple','test_recall_simple'
                                  ,'valid_specificity_weighed','test_specificity_weighed'
                                  ,'valid_specificity_simple','test_specificity_simple','train_index','valid_index'])
    
    bar = progressbar()
    grid_params = sk.model_selection.ParameterGrid(grid)
    if cross_valid:
        total_iter = nr_repeat*len(grid_params)*kflods.get_n_splits()
    else: total_iter = nr_repeat*len(grid_params)
    iter = 1

    max_test_accuracy = 0
    max_test_fscore = 0
    max_valid_accuracy = 0
    max_valid_fscore = 0

    for i in range(nr_repeat):
        for g in grid_params:

            train_index = None
            valid_index = None
            if cross_valid:
                training_data = np.concatenate((X_train,X_valid ))
                training_label = np.concatenate((Y_train,Y_valid ))
                
                for fold, (train_index, valid_index) in enumerate(kflods.split(training_data,training_label)):
                    bar.bar(iter,total_iter, size=50)
                    iter +=1
                    x_train_ = training_data[train_index]
                    y_train_ = training_label[train_index]
                    x_valid_ = training_data[valid_index]
                    y_valid_ = training_label[valid_index]
                    
                    Model_.set_params(**g)
                    Model_.fit(x_train_, y_train_)

                    # Valid
                    Y_pred = Model_.predict(x_valid_)
                    Y = y_valid_
                    valid_recall_weighted, valid_specificity_weighed, _ = recall_specificity_precision_fscore(Y,Y_pred,weighted_avg=True)
                    
                    valid_recall_simple, valid_specificity_simple, _ = recall_specificity_precision_fscore(Y,Y_pred,weighted_avg=False)
                    # Test
                    Y_pred = Model_.predict(X_test)
                    Y = Y_test
                    test_recall_weighted, test_specificity_weighed, _ = recall_specificity_precision_fscore(Y,Y_pred,weighted_avg=True)
                    test_recall_simple, test_specificity_simple, _ = recall_specificity_precision_fscore(Y,Y_pred,weighted_avg=False)

                    new_row = pd.Series({'hyper_param':g,'valid_recall_weighted':valid_recall_weighted,'test_recall_weighted':test_recall_weighted
                                        ,'valid_recall_simple':valid_recall_simple,'test_recall_simple':test_recall_simple
                                        ,'valid_specificity_weighed':valid_specificity_weighed,'test_specificity_weighed':test_specificity_weighed
                                        ,'valid_specificity_simple':valid_specificity_simple,'test_specificity_simple':test_specificity_simple,'train_index':train_index,'valid_index':valid_index}, name='')
                    Param_Details = Param_Details.append(new_row)

                    if valid_recall_weighted > max_valid_accuracy: 
                        max_valid_accuracy = valid_recall_weighted
                        Expected_Test_Acc = test_recall_weighted
                        pickle.dump(Model_ , open('Best_Model','wb'))
                    if test_recall_weighted > max_test_accuracy: 
                        max_test_accuracy = test_recall_weighted                        
                    bar.bar(iter,total_iter, size=30)
                    bar.set_pre_description(f'Max_Valid_Acc.: {max_valid_accuracy:.1%}, Max_Test_Acc.: {max_test_accuracy:.1%}, Expected_test_Acc.: {Expected_Test_Acc:.1%}')

            else:
                    
                    iter +=1
                    x_train_ = X_train
                    y_train_ = Y_train
                    x_valid_ = X_valid
                    y_valid_ = Y_valid
                    
                    Model_.set_params(**g)
                    Model_.fit(x_train_, y_train_)

                    # Valid
                    Y_pred = Model_.predict(x_valid_)
                    Y = y_valid_
                    valid_recall_weighted, valid_specificity_weighed, _ = recall_specificity_precision_fscore(Y,Y_pred,weighted_avg=True)
                    valid_recall_simple, valid_specificity_simple, _ = recall_specificity_precision_fscore(Y,Y_pred,weighted_avg=False)
                    # Test
                    Y_pred = Model_.predict(X_test)
                    Y = Y_test
                    test_recall_weighted, test_specificity_weighed, _ = recall_specificity_precision_fscore(Y,Y_pred,weighted_avg=True)
                    test_recall_simple, test_specificity_simple, _ = recall_specificity_precision_fscore(Y,Y_pred,weighted_avg=False)

                    new_row = pd.Series({'hyper_param':g,'valid_recall_weighted':valid_recall_weighted,'test_recall_weighted':test_recall_weighted
                                        ,'valid_recall_simple':valid_recall_simple,'test_recall_simple':test_recall_simple
                                        ,'valid_specificity_weighed':valid_specificity_weighed,'test_specificity_weighed':test_specificity_weighed
                                        ,'valid_specificity_simple':valid_specificity_simple,'test_specificity_simple':test_specificity_simple,'train_index':train_index,'valid_index':valid_index}, name='')
                    Param_Details = Param_Details.append(new_row)

                    if valid_recall_weighted > max_valid_accuracy: 
                        max_valid_accuracy = valid_recall_weighted
                        Expected_Test_Acc = test_recall_weighted
                        pickle.dump(Model_ , open('Best_Model','wb'))
                    if test_recall_weighted > max_test_accuracy: 
                        max_test_accuracy = test_recall_weighted                        
                    bar.bar(iter,total_iter, size=30)
                    bar.set_pre_description(f'Max_Valid_Acc.: {max_valid_accuracy:.1%}, Max_Test_Acc.: {max_test_accuracy:.1%}, Expected_test_Acc.: {Expected_Test_Acc:.1%}')

    Best_Model = pickle.load(open('Best_Model','rb'))
   
    best_one = np.argmax(Param_Details.valid_recall_weighted)
    best_param = Param_Details.iloc[best_one]['hyper_param']
    # best_train_index = Param_Details.iloc[best_one]['train_index'] 
    # best_valid_index = Param_Details.iloc[best_one]['valid_index']  
    # if cross_valid:
    #         x_train_ = training_data[best_train_index]
    #         y_train_ = training_label[best_train_index]
    #         x_valid_ = training_data[best_valid_index]
    #         y_valid_ = training_label[best_valid_index]

    # Best_Model = Model_.set_params(**best_param)
    # Best_Model.fit(x_train_, y_train_)
    Y_pred = Best_Model.predict(X_test)
    Param_Details['hyper_param'] = Param_Details['hyper_param'].apply(lambda x: json.dumps(x))
    print('\n------- Precision recal %--------')
    print(sk.metrics.classification_report(Y_test,Y_pred))
    print('Best param: ' , best_param)
    return Best_Model, Param_Details

class progressbar:
  def __init__(self):
    self.pre = ' '
    self.post1 = ' '
    self.post2 = ' '
    self.start_time = time.time()
    self.init_time = time.time()
  def set_pre_description(self,s):
    self.pre = s
  def set_first_description(self,s):
    self.post1 = s
  def set_second_description(self,s):
    self.post2 = s
  def bar(self,n,tot, prefix="", size=30, out=sys.stdout, show_time=True):
      current_time = time.time()
      time_cost = current_time - self.init_time
      total_time = current_time -self.start_time 
      if n==0:
        remaining_time = (tot-n)/(n+1) * total_time
      else:
        remaining_time = (tot-n)/(n) * total_time
      self.init_time = time.time()
      count = tot
      def show(j):
          x = int(size*j/count)
          if show_time:
            out.write(f'\r {self.pre} [{u"█"*x}{"."*(size-x)}] {j}/{count} [Time => Iter.: {time_cost:.1f}s, Tot.: {total_time:.1f}s, Remain.: {remaining_time:.1f}s] {self.post1} {self.post2}')
          else:
            out.write(f'\r {self.pre} [{u"█"*x}{"."*(size-x)}] {j}/{count}  {self.post1} {self.post2}')
          out.flush()        
      show(0)
      show(n)
      out.flush()

def plot_cnn_loss_accuracy(model_):
    epochs_X = [i for i in range(1, len(model_.Epochs_Train_loss)+1)]
    fig, axs = plt.subplots(1,2,figsize=(14,4))
    axs[0].plot(epochs_X , model_.Epochs_Train_loss , 'bo-', label='Train loss')
    axs[0].plot(epochs_X , model_.Epochs_Val_loss,'ro-', label='Validation loss')
    axs[0].plot(epochs_X , model_.Epochs_test_loss,'go-', label='Test loss')
    axs[0].set_xlabel("Epochs", fontsize = 12)
    axs[0].set_ylabel("Loss", fontsize = 12)
    axs[0].grid()
    axs[0].legend()
    axs[0].set_title('Train and Validation loss by epochs', fontsize = 14)
    axs[1].plot(epochs_X , model_.Epochs_Train_Acc , 'bo-', label='Train Accuracy')
    axs[1].plot(epochs_X , model_.Epochs_Val_Acc ,'ro-', label='Validation Accuracy')
    axs[1].plot(epochs_X , model_.Epochs_test_Acc ,'go-', label='Test Accuracy')
    axs[1].set_xlabel("Epochs", fontsize = 12)
    axs[1].set_ylabel("Accuracy", fontsize = 12)
    axs[1].grid()
    axs[1].legend()
    axs[1].set_title('Train and Validation Accuracy by epochs', fontsize = 14)
    plt.show()

def recall_specificity_precision_fscore(Y,Y_pred, weighted_avg):
    CM = sk.metrics.confusion_matrix(Y,Y_pred)
    FP = CM.sum(axis=0) - np.diag(CM) 
    FN = CM.sum(axis=1) - np.diag(CM)
    TP = np.diag(CM)
    TN = CM.sum() - (FP + FN + TP)
    weights = CM.sum(axis=1) / CM.sum() 
    Accuracy = np.nan_to_num((TP+TN)/(TP+FP+FN+TN) , nan=0)
    Recall = np.nan_to_num(TP/(TP+FN) , nan=0)
    Specificity = np.nan_to_num(TN/(TN+FP) , nan=0)
    Precision = np.nan_to_num(TP/(TP+FP) , nan=0)
    if weighted_avg:
        return round(sum(weights*Recall),3), round(sum(weights*Specificity),3), round(sum(weights*Precision),3)
    else:
        return round(mean(Recall),3), round(mean(Specificity),3), round(mean(Precision),3)