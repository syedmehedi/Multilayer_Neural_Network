# -*- coding: utf-8 -*-
"""
Created on Thu May  2 09:00:39 2019

@author: syedm
"""
import h5py  
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import datetime
from datetime import datetime


class Activation(object):
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        # a = np.tanh(x)   
        return 1.0 - a**2
    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_deriv(self, a):
        # a = logistic(x) 
        return  a * (1 - a )
    
    def __init__(self,activation='tanh'):
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
        elif activation == 'softmax':
            self.f = self.__softmax
            self.f_deriv = self.__softmax_deriv
        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv
            
            
    # Activation function to get probabilty output from output laer
    def __softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference
     # Derivative of softmax
    def __softmax_deriv(self, a):
        """Compute softmax values for each sets of scores in x."""
        return self.__softmax(a)*(1-self.__softmax(a))
    
    #Activation function relu
    def __relu(self, x):
        x=np.maximum(0,x)
        return x
    # Derivative of relu
    def __relu_deriv(self, x):   
        x=np.maximum(0,1) 
        return x
    
    
# ### Custom Functions


# ### Define HiddenLayer

class HiddenLayer(object):    
    def __init__(self,n_in, n_out,
                 activation_last_layer='softmax',activation='tanh', W=None, b=None, v_W=None, v_b=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input=None
        self.activation=Activation(activation).f
        
        # activation deriv of last layer
        self.activation_deriv=None
        if activation_last_layer:
            self.activation_deriv=Activation(activation_last_layer).f_deriv

        self.W = np.random.uniform(
                low=-np.sqrt(5. / (n_in + n_out)),
                high=np.sqrt(5. / (n_in + n_out)),
                size=(n_in, n_out)
        )
        if activation == 'logistic':
            self.W *= 4

        self.b = np.zeros(n_out,)
        
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)
        
    def forward(self, input):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''
        lin_output = np.dot(input, self.W) + self.b
               
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input=input
       
        return self.output
    
    #forward with drop out rate provided
    def forward_dropout(self, input,drop_out=0.25):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''
        lin_output = np.dot(input, self.W) + self.b
        #drop fraction of neurons provided by var drop_out
        lin_output=self.drop_out(lin_output, drop_out)    
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input=input       
        return self.output
    
    #back propagate delta to update weigth and biases
    def backward(self, delta, output_layer=False): 
        
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta)) 
        self.grad_b = delta
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
            
        return delta
    
    
    #Drop percentage of nodes according to drop_out percentage provided
    def drop_out(self, input_mat, drop_out=0.25):
        D =  np.random.rand(*input_mat.shape)             
        D = D <   (1-drop_out )                                     
        input_mat = input_mat * D                                               
        input_mat = input_mat / (1-drop_out )
        
        return input_mat
    
#---------------------------------------   The MLP--------------
# ## The MLP
# The class implements a MLP with a fully configurable number of layers and neurons. It adapts its weights using the backpropagation algorithm in an online manner.
class MLP:
    """
    """      
    def __init__(self,layers, test_data, test_label, activation=[None,'tanh','softmax'], Drop_Out=0.25, Weight_Decay=0.003, batch_size=100, Momentum=0.9):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """        
        ### initialize layers
        self.layers=[]
        self.params=[]
        
        self.activation=activation
        self.batch_size=batch_size
        self.Drop_Out=Drop_Out
        self.Weight_Decay=Weight_Decay
        self.Momentum=Momentum
        self.test_data=test_data
        self.test_label=test_label
        
        
        for i in range(len(layers)-1):
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1]))
            
    def forward(self,input, Drop_Out):
        n_layer=0 #me
        for layer in self.layers:          
            if (n_layer==1 and self.Drop_Out!=None): #me
                output=layer.forward_dropout(input, Drop_Out)
            #elif n_layer==3: 
                #output=layer.forward_dropout(input)
            else:
                output=layer.forward(input)
            input=output
            
            n_layer +=1 #me
        return output
    
 
    #softmax activation function
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference
    
    
    #Calculate cross entropy loss from predicted and actual label
    def cross_entropy_loss(self, y,y_hat):
        
        activation_deriv=Activation(self.activation[-1]).f_deriv
        delta=-(y-y_hat)*activation_deriv(y_hat) 
        return np.sum(-y * np.log(1e-15 + y_hat)),delta
    
    #calculate gradient of cross_entropy
    def delta_cross_entropy(self,X,y):
   
       # X is the output from fully connected layer (num_examples x num_classes)
       # y is labels (num_examples x 1)
       # 	Note that y is not one-hot encoded vector. 
          
        m = y.shape[0]
        grad = self.softmax(X)
        grad[range(m),y] -= 1
        grad = grad/m
        return grad
    
    
    
    
    def backward(self,delta):
        
        delta=self.layers[-1].backward(delta,output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta=layer.backward(delta)
   
    #update weight and bias using Momentum (gamma)
    def update(self,lr, n_epochs, gamma=0.8):
        
        for layer in self.layers:
            #Initialize gradient of weight and bias as 0 in first epoch
            if n_epochs == 0:              
                layer.v_W=np.zeros_like(layer.grad_W) 
                layer.v_b=np.zeros_like(layer.grad_b) 
            #update gradient using Momentum(gamma) and learning rate           
            layer.v_W = gamma * layer.v_W + lr * layer.grad_W
            layer.v_b = gamma * layer.v_b + lr * layer.grad_b
            
            layer.W -= layer.v_W
            layer.b -= layer.v_b
     
    #Update weight and bias using Weight Decay(L2 regularisation) and Momentum (gamma) 
    def update_With_L2(self,lr, n_epochs, lamda=0.0003, gamma=0.65, Weight_Decay=0.003):
        
        for layer in self.layers:
            #Initialize gradient of weight and bias as 0 in first epoch
            if n_epochs == 0:               
                layer.v_W=np.zeros_like(layer.grad_W) 
                layer.v_b=np.zeros_like(layer.grad_b) 
            #update gradient using Momentum(gamma) and learning rate   
            layer.v_W = gamma * layer.v_W + lr * layer.grad_W
            layer.v_b = gamma * layer.v_b + lr * layer.grad_b
            
            layer.W=(1-lamda * lr) * layer.W -layer.v_W
            layer.b=(1-lamda * lr) * layer.b -layer.v_b
           
            
            
    def fit(self,X,y,test_data,test_label,learning_rate=0.001, epochs=100, batch_size=100, Drop_Out=0.25):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """ 
        X=np.array(X)
        y=np.array(y)
        to_return = np.zeros(epochs)
        
        validation_accuracy=0
        val_accuracy=[]#np.zeros(epochs)
        
        acc_increase=True
        number_decrease=0
                        
        str_time_start=str(datetime.now())
        str_time_start=str_time_start[:str_time_start.index('.')]
        
        #Continue as long as Validation accuracy keep increasing
        while acc_increase==True:
           
            for k in range(epochs):
               
                loss_SGD=self.mini_batch_SGD(X, y, k, learning_rate, batch_size)
                
                to_return[k] = loss_SGD #np.mean(loss)
                
                #calculate validation accuracy
                if ((k % 5)==0):
                  preds_test = self.predict(test_data,Drop_Out)
                  acc=sum(preds_test==test_label)/len(test_label)
                                   
                  val_accuracy.append(acc)
                  
                
                if ((k % 5)==0):
                    print('Epochs :' + str(k+1) + ' of ' + str(epochs) + ' Loss: ' + str(loss_SGD)[:8] +  ", Validation Accuracy: ",  str(acc) if len(str(acc)) <=4 else str(acc)[:8])
                
                    if validation_accuracy<acc:
                        validation_accuracy=acc
                        number_decrease=0
                    else:
                        number_decrease +=1
                        validation_accuracy=acc
                        #if conseutive two times accuracy decraese on validation data
                        if (number_decrease>=2):
                            acc_increase=False
                            #print("Previous Accuracy:" + str(validation_accuracy) + "current:" + str(acc))
                            to_return=to_return[:k]
                            str_time_later=str(datetime.now())
                            str_time_later=str_time_later[:str_time_later.index('.')]
                            fmt = '%Y-%m-%d %H:%M:%S'
                            d1 = datetime.strptime(str_time_start, fmt)
                            d2 = datetime.strptime(str_time_later, fmt)
                            time_spent=d2-d1
                            break
                if k==epochs-1:
                    acc_increase=False
                    to_return=to_return[:k]
                    str_time_later=str(datetime.now())
                    str_time_later=str_time_later[:str_time_later.index('.')]
                    fmt = '%Y-%m-%d %H:%M:%S'
                    d1 = datetime.strptime(str_time_start, fmt)
                    d2 = datetime.strptime(str_time_later, fmt)
                    time_spent=d2-d1
                
        return to_return, val_accuracy,time_spent

    
    #Predict label using forward function of our model
    def predict(self, x,Drop_Out):
        x = np.array(x)
        output = np.zeros((x.shape[0],10))
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i,:],Drop_Out)
        
        predict_class = np.argmax(output, axis=1)
        predict_class = predict_class.tolist()
        predict_class
        return predict_class
    
    #Calculate loss and gradient using mini-batch.
    def mini_batch_SGD(self, X, y, n_epochs, learning_rate=0.1, batch_size=16, Drop_Out=0.25, Weight_Decay=0.003, Momentum=0.9):
        
        m=len(y)
        n_batches=int(m/batch_size)
        indices=np.random.permutation(m)
        X, y=X[indices], y[indices]        
        loss=np.zeros(n_batches) # 
        num_loss=0
        # Take a mini-bacth at a time 
        for j in range(0,m,batch_size):  
            X_j, y_j=X[j:j+batch_size], y[j:j+batch_size]
            i=np.random.randint(X_j.shape[0])
            m=0
            loss_m=np.zeros(batch_size)
            delta=np.zeros(10)            
            #calculate delta and loss for mini-batch
            for _ in range(batch_size):   
                y_hat = self.forward(X_j[i], Drop_Out)         
                loss_m[m], delta_m=self.cross_entropy_loss(y_j[i],y_hat)
                delta +=delta_m
                m +=1
            #calculate mean loss of a mini-batch
            loss[num_loss]= np.mean(loss_m)         
            num_loss +=1          
            #pass average delta of the mini-batch to the backward function
            self.backward(delta/batch_size)
            #Drop_Out=0.25, Weight_Decay=0.003
            #update weight and bias
            if self.Weight_Decay!=None:     
              self.update_With_L2(learning_rate, n_epochs,lamda=Weight_Decay, gamma=Momentum) #by me for momentum
            else:
              self.update(learning_rate, n_epochs, gamma=Momentum)
            
        
        mean_loss=np.mean(loss)
        return mean_loss
        