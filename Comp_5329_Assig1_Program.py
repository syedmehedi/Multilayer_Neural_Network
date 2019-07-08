
# # COMP5329 - Deep Learning - Assignment-1

# ## 

# ### Based on the codes given in Tutorial: Multilayer Neural Network, you are
# ### required to accomplish a multi-class classification task on the provided
# ### dataset.

# ## Submitted by:
# ### Syed Mehedi Hasan (Student Id: 480255897, Unikey: shas5428)
# ### Mudassar Riaz(Student Id: 460238922, Unikey: mria6883)
#### Xiaodong Zhao (490373431)

# ## Readme
# ###How to Run:
# From Spyder
#### 1. Please open code/Alogithm/Comp_5329_Assig1_Program.py 
# #### 2. Please goto  Menu -> Run ->  or from keyboard please press (ctrl + F5)

# From Anaconda Prompt
###     1.Please change directory to Code/Alogorithm then
# #### 2. Please type python Comp_5329_Assig1_Program.py
#Our functions and class import
from our_functions import One_hot_label,f1_score, write_label_h5, get_confusion_matrix, heatmap_confusion_matrix, plot_graph, plot_accuracy, show_label_image
from Our_MLP import MLP, Activation, HiddenLayer

# ## Loading the packages
import h5py  
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

##Load Train data
with h5py.File('../Input/train_128.h5','r') as H:
    data = np.copy(H['data'])
    
##Load Train label    
with h5py.File('../Input/train_label.h5','r') as H:
    label = np.copy(H['label'])
    
##Load Test data
with h5py.File('../Input/test_128.h5','r') as H:
    final_test_data= np.copy(H['data'])

#Total number of unique label
n_label=np.unique(label)
n_col=data.shape[1]

# ### Join Data and label and Shuffle
data_set=np.hstack((data, label.reshape(len(label),1)))
m=len(data_set)

indices=np.random.permutation(m)
data_set_new=data_set[indices]

# ### Show frequncy distribution of each category of label using bar chart
df_train_label=pd.DataFrame(label)
df_train_label.columns = ['label']
labels = sorted(df_train_label['label'].unique())

df_train_label['label'].value_counts().plot(kind='bar', color=['Orange','green','blue','black','yellow','magenta','cyan','purple','gold','olive'])

# ## Visualise some train label of each category
show_label_image(pl, data, label)

# ### Train Test split  and Scaling data by dividing 255 (as it looks image data)

#Train and test split  (80% train and 20% test)    
split_train = int(0.8 * len(data_set_new))
##Scale train, validation and test data by diving 255
train_data=data_set_new[:split_train,:n_col]/255
train_label=data_set_new[:split_train,n_col].reshape(split_train,1)
test_data=data_set_new[split_train:,:n_col]/255
test_label=data_set_new[split_train:,n_col]
print('Shapes of Train and Validation Data:')
print(train_data.shape,train_label.shape, test_data.shape,test_label.shape)

final_test_data=final_test_data/255

#Join train data and train label in same array to avoid index mixup
data_set_new=np.hstack((train_data, train_label))
data_set_new.shape
data_set_new[:-100,n_col:n_col+1]
data_set_new.shape
dataset=data_set_new

#Please Set Hyperparameter here:
BATCH_SIZE = 100 
LEARNING_RATE = 0.0017
WEIGHT_DECAY =0.0003 #or None if you do not want
DROPOUT_RATE = 0.25 # 0.25, None if you do not want 
MOMENTUM=0.9
N_EPOCHS=100

### Run Multilayer Neural Network with our hyper parameter provided here
nn = MLP(test_data=test_data,test_label=test_label, Drop_Out=None, Weight_Decay=WEIGHT_DECAY, Momentum=MOMENTUM, layers=[128,128,64,10], activation=[None,'relu', 'tanh', 'softmax'])
input_data = dataset[:,0:n_col]
labels=dataset[:,n_col]
#make one hot vector from label
output_data = One_hot_label(dataset[:,n_col])

MSE, val_acc,time_spent = nn.fit(input_data, output_data,test_data,test_label, learning_rate=LEARNING_RATE, epochs=N_EPOCHS, batch_size=BATCH_SIZE)

print("Time taken for training Model:" + str(time_spent))
#Plot training loss vs Number of epochs
plot_graph(pl,MSE,'Number of Epochs','Loss', 'Loss vs Epochs during training')
#Plot Validation Accuracy vs Number of epochs
plot_accuracy(pl,[x*5 for x in range(len(val_acc))], val_acc,'Number of Epochs','Model Accuracy', 'Validation Accuracy vs Epochs during Model training')

# ## Predict label using model and calculate accuracy

#Validation Accuracy
Preds_Test = nn.predict(test_data, DROPOUT_RATE)
acc=sum(Preds_Test==test_label)
print("Validation Accuracy for Validation data: ", acc/len(test_label))

#Train Accuracy
Preds_Train = nn.predict(input_data, DROPOUT_RATE)
acc=sum(Preds_Train==labels)
print("Train Accuracy for training data: ", acc/len(labels))

#Predict our Original Test Data
preds=nn.predict(final_test_data, DROPOUT_RATE)

df=pd.DataFrame(preds, columns=['cls'])

#Write our predicted label of Original Test data in Output Folder with unique name each run
write_label_h5(preds)

conf_matrix=get_confusion_matrix(Preds_Test, test_label)

# Calculate Precision, Recall and F1 Score
lst_Pres,lst_Recall,lst_f1_score=f1_score(conf_matrix)

df_f1_score=pd.DataFrame({' Label':[x for x in range(len(n_label))],' Precision':lst_Pres,' Recall':lst_Recall,'F1_Score':lst_f1_score })
print(df_f1_score.to_string(index=False))

#show confusion matrix as heatmap #
heat_map=heatmap_confusion_matrix(conf_matrix)



