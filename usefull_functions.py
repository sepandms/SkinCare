
import sklearn as sk
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
import numpy as np
from tqdm import tqdm
import json

def model_evaluation(Y,Y_pred,title='Confusion Matrix'):
    CM = sk.metrics.confusion_matrix(Y,Y_pred)
    print('Nr. of Data : \n', CM.sum())
    print('Accuracy of The Model : \n', np.diag(CM).sum()/CM.sum())
    disp = sk.metrics.ConfusionMatrixDisplay(CM)
    disp.plot()
    disp.im_.colorbar.remove()
    font = {'family' : 'normal','size'   : 14}
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


def recall_specificity_precision(Y,Y_pred, weighted_avg):
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


def progressbar(n,tot, prefix="", size=60, out=sys.stdout):
    count = tot
    def show(j):
        x = int(size*j/count)
        out.write("%s[%s%s] %i/%i\r" % (prefix, u"█"*x, "."*(size-x), j, count))
        out.flush()        
    show(0)
    show(n)
    out.flush()

def grid_searc_cross_valid_trainer(Model_, grid, cross_valid, kflods ,X_train , Y_train,X_valid,Y_valid,X_test, Y_test,nr_repeat ):

    Param_Details = pd.DataFrame(columns=['hyper_param','valid_recall_weighed','test_recall_weighed'
                                  ,'valid_recall_simple','test_recall_simple'
                                  ,'valid_specificity_weighed','test_specificity_weighed'
                                  ,'valid_specificity_simple','test_specificity_simple','train_index','valid_index'])
    
    grid_params = sk.model_selection.ParameterGrid(grid)
    total_iter = nr_repeat*len(grid_params)*kflods.get_n_splits()
    iter = 1
    for i in range(nr_repeat):
        for g in grid_params:
            test_accuracy = []
            test_fscore = []
            test_precision = []
            test_recall = []
            train_index = None
            valid_index = None
            if cross_valid:
                training_data = np.concatenate((X_train,X_valid ))
                training_label = np.concatenate((Y_train,Y_valid ))
                
                for fold, (train_index, valid_index) in enumerate(kflods.split(training_data,training_label)):
                    progressbar(iter,total_iter, "Interations: ", 100)
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
                    valid_recall_weighed, valid_specificity_weighed, _ = recall_specificity_precision(Y,Y_pred,weighted_avg=True)
                    valid_recall_simple, valid_specificity_simple, _ = recall_specificity_precision(Y,Y_pred,weighted_avg=False)
                    # Test
                    Y_pred = Model_.predict(X_test)
                    Y = Y_test
                    test_recall_weighed, test_specificity_weighed, _ = recall_specificity_precision(Y,Y_pred,weighted_avg=True)
                    test_recall_simple, test_specificity_simple, _ = recall_specificity_precision(Y,Y_pred,weighted_avg=False)

                    new_row = pd.Series({'hyper_param':g,'valid_recall_weighed':valid_recall_weighed,'test_recall_weighed':test_recall_weighed
                                        ,'valid_recall_simple':valid_recall_simple,'test_recall_simple':test_recall_simple
                                        ,'valid_specificity_weighed':valid_specificity_weighed,'test_specificity_weighed':test_specificity_weighed
                                        ,'valid_specificity_simple':valid_specificity_simple,'test_specificity_simple':test_specificity_simple,'train_index':train_index,'valid_index':valid_index}, name='')
                    Param_Details = Param_Details.append(new_row)
            else:
                    progressbar(iter,total_iter, "Interations: ", 100)
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
                    valid_recall_weighed, valid_specificity_weighed, _ = recall_specificity_precision(Y,Y_pred,weighted_avg=True)
                    valid_recall_simple, valid_specificity_simple, _ = recall_specificity_precision(Y,Y_pred,weighted_avg=False)
                    # Test
                    Y_pred = Model_.predict(X_test)
                    Y = Y_test
                    test_recall_weighed, test_specificity_weighed, _ = recall_specificity_precision(Y,Y_pred,weighted_avg=True)
                    test_recall_simple, test_specificity_simple, _ = recall_specificity_precision(Y,Y_pred,weighted_avg=False)

                    new_row = pd.Series({'hyper_param':g,'valid_recall_weighed':valid_recall_weighed,'test_recall_weighed':test_recall_weighed
                                        ,'valid_recall_simple':valid_recall_simple,'test_recall_simple':test_recall_simple
                                        ,'valid_specificity_weighed':valid_specificity_weighed,'test_specificity_weighed':test_specificity_weighed
                                        ,'valid_specificity_simple':valid_specificity_simple,'test_specificity_simple':test_specificity_simple,'train_index':train_index,'valid_index':valid_index}, name='')
                    Param_Details = Param_Details.append(new_row)


    best_one = np.argmax(Param_Details.test_recall_weighed)
    best_param = Param_Details.iloc[best_one]['hyper_param']
    best_train_index = Param_Details.iloc[best_one]['train_index'] 
    best_valid_index = Param_Details.iloc[best_one]['valid_index']  
    if cross_valid:
            x_train_ = training_data[best_train_index]
            y_train_ = training_label[best_train_index]
            x_valid_ = training_data[best_valid_index]
            y_valid_ = training_label[best_valid_index]

    Best_Model = Model_.set_params(**best_param)
    Best_Model.fit(x_train_, y_train_)
    Y_pred = Best_Model.predict(X_test)
    Param_Details['hyper_param'] = Param_Details['hyper_param'].apply(lambda x: json.dumps(x))
    print('------- Precision recal %--------')
    print(sk.metrics.classification_report(Y_test,Y_pred))
    print('Best param: ' , best_param)
    return Best_Model, Param_Details