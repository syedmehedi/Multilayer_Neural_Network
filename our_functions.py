# -*- coding: utf-8 -*-
"""
Created on Thu May  2 08:33:24 2019

@author: syedm
"""
import numpy as np
import pandas as pd
import h5py  
from datetime import datetime
#import matplotlib.pyplot as pl
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

#Make one hot label from given label
def One_hot_label(label):
    
    label=np.rint(label)
    n=len(label)
    m_class=10
        
    one_hot_labels = np.zeros((n, m_class))
    
    for i in range(n):  
        one_hot_labels[i, int(label[i])] = 1
    
    return one_hot_labels

#Calculate Precision, Recall and F1 Score of all class labels
def f1_score(conf_matrix):
  
    lst_FP=[]
    lst_FN=[]
    lst_TP=[]
    lst_Pres=[]
    lst_Recall=[]
    lst_f1_score=[]
    for k in range(len(conf_matrix)):
      FP=sum([conf_matrix[k][i] for i in range(len(conf_matrix))]) - conf_matrix[k][k]
      lst_FP.append(FP)
      lst_TP.append(conf_matrix[k][k])
    
    for k in range(len(conf_matrix)):
      FN=sum([conf_matrix[i][k] for i in range(len(conf_matrix))]) - conf_matrix[k][k]
      lst_FN.append(FN)
      
    for i in range(len(conf_matrix)):
      pres=lst_TP[i]/(lst_TP[i]+lst_FP[i])
      recall=lst_TP[i]/(lst_TP[i]+lst_FN[i])
      
      lst_Pres.append(pres)
      lst_Recall.append(recall)
      
    for i in range(len(conf_matrix)):
      f1_score=((lst_Pres[i]*lst_Recall[i])/(lst_Pres[i] + lst_Recall[i]))*2
      lst_f1_score.append(f1_score)
      
    return lst_Pres, lst_Recall,lst_f1_score

#write as h5File  to output folder 
def write_label_h5(preds):
    
    time_stamp = datetime.now().timestamp()
    str_time_stamp=str(int(time_stamp))

    #make unique file_name using timestapm()     
    file_name='../Output/Predicted_Label_' + "_" + str_time_stamp + '.h5'
    file_path_output_data = pathlib.Path(__file__).parent / file_name
    
    hf = h5py.File(file_path_output_data, 'w')
    hf.create_dataset('label', data=preds)
    hf.close()
    
#calculate and show confustion matrix
def get_confusion_matrix(prediction, actual):
    conf_matrix = pd.DataFrame(list(zip(prediction,actual)), 
                                columns=['predicted labels','actual labels'])
    conf_matrix['const'] = 1
    conf_matrix = pd.pivot_table(data=conf_matrix, 
                               index='actual labels', 
                               columns='predicted labels', 
                               values='const', 
                               aggfunc=sum)
    conf_matrix = conf_matrix.fillna(0)
    return conf_matrix

#Draw heat map from confusion matrix
def heatmap_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8,6))
    g = sns.heatmap(conf_matrix, cbar_kws={'label':'Frequency'}, center=0, cmap=sns.diverging_palette(10, 5, as_cmap=True)).set_title('Confusion Matrix')
    return g

#Draw Loss vs Number of Epochs
def plot_graph(plt1, data, xlab, ylab, title):
    plt1.figure(figsize=(15,4))
    plt1.figure(figsize=(15,4))
    plt1.plot(data)
    plt1.xlabel(xlab)
    plt1.ylabel(ylab)
    plt1.title(title)
    plt1.grid()
    return plt1

#draw accuracy vs Number of Epochs
def plot_accuracy(plt1, x,y, xlab, ylab, title):
    plt1.figure(figsize=(15,4))
    plt1.figure(figsize=(15,4))
    plt1.plot(x,y)
    plt1.xlabel(xlab)
    plt1.ylabel(ylab)
    plt1.title(title)
    plt1.grid()
    return plt1

#Show one picture for each label
def show_label_image(plt1,data, label):
    dict_img={}
    lst_label=[]

    for i in range(100):
      if label[i] not in lst_label:
          lst_label.append(label[i])
          dict_img[label[i]]=data[i]
          
      if(len(lst_label)==10):
        break
            
    #using pyplot shoaw each distinct label as original color image
    plt1.figure(figsize=[12,12])
    i=0
    for key in sorted(dict_img.keys()):         
          plt1.subplot(6,5,i+1)
          plt1.title("Label: %i"%key)
          img=dict_img[key]
          plt1.imshow(img.reshape([8,16]));
          i +=1
          
    return plt1