#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

data = pd.read_csv('Assignment_4_data.txt', sep='\t',header=None).values

x = data[:,1]
y = data[:,0]

import nltk
nltk.download('stopwords')
stop_words = (set(stopwords.words("english")))


# List of lists of words
# Removes punctuations, converts to everything to lowercase
# Tokenization complete
# Stopword removal also done using nltk.stopwords
PS = PorterStemmer()
words = []
sentences = []
vocab = []
tokenizer = RegexpTokenizer(r'\w+')
i = 0
for sent in x:
    for word in tokenizer.tokenize(sent):     # Tokenization
        if(word not in stop_words):           # Checking for stopwords
            word = PS.stem(word)              # Applying Porter Stemming
            words.append(word.lower())
            vocab.append(word.lower())
    sentences.append(words)
    words = []



num_sentences = len(sentences)
vocab_size = len(set(vocab))
print(vocab_size)
print(num_sentences)
#print(set(vocab))

vocab = list(set(vocab))
#print(vocab)


# In[8]:


word_2_idx = {}
i = 0
for word in vocab:
    word_2_idx[word] = i
    i+=1



# One-hot vector generation for all sentences
X = np.zeros((num_sentences,vocab_size))
for i in range(0,len(sentences)):
    for j in range(0,len(sentences[i])):
        word = sentences[i][j]
        idx = word_2_idx[word]
        X[i][idx] = 1



le = LabelEncoder()
le.fit(y)
y = le.transform(y)

# Neural Network Specifics
# num_hidden_layers = 1
# num_nodes_h1 = 100
# ReLU in h1
# Output layer - 1 neuron
# Optimization algorithm: SGD
# Loss fucntion: Categorical Cross Entropy Loss


# In[201]:


def ReLU(x):
    f = np.maximum(0,x)
    return f


# In[202]:


def ReLU_grad(x):
    grad = np.zeros(np.shape(x))
    for i in range(0,np.shape(x)[0]):
        for j in range(0,np.shape(x)[1]):
            if(x[i][j]>0):
                grad[i][j] = 1
            else:
                grad[i][j] = 0
    return grad


# In[203]:


def sigmoid(x):
    #print(np.shape(x))
    sig = np.exp(x)/(1 + np.exp(x))
    #print(np.shape(sig))
    return sig


# In[204]:


def sigmoid_grad(sig):
    '''Takes value of sigmoid as input and returns gradient w.r.t x '''
    # x is not taken as an input
    return sig*(1-sig)


# Dimensions of Neural Network

input_dim = vocab_size  # input dimension
hidden_1_dim = 100  # hidden layer 1 dimension
output_dim = 1   # output dimension

# Other hyperparameters 
learning_rate = 0.1


# In[324]:


def Weight_initialiser():
    W1 = np.random.uniform(size=(hidden_1_dim,input_dim),low=-1,high=1)    # 100*7375
    W2 = np.random.uniform(size=(output_dim,hidden_1_dim),low=-1,high=1)   # 1*100
#     W1 = np.random.normal(size=(hidden_1_dim,input_dim))    # 100*7375
#     W2 = np.random.normal(size=(output_dim,hidden_1_dim))   # 1*100
    return [W1,W2]


# In[384]:


def random_mini_batches(X, Y, mini_batch_size=32, seed = 0):
    np.random.seed(seed)            
    m = X.shape[0]                  # number of training examples
    mini_batches = []
   
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation]
    
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k*mini_batch_size:(k+1)*mini_batch_size]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches*mini_batch_size:m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches*mini_batch_size:m]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# In[385]:


def forward(X,W):
    # X: input matrix (batch_size x 7375)
    W1,W2 = W
    # W1: (100 x 7375)
    # W2: (1 x 100)
    #print(np.shape(W1),np.shape(W2))
    # First hidden layer and activation layer
    h1 = np.matmul(X,np.transpose(W1))    # (batch_size x 100)
    a1 = ReLU(h1)
    
    # Output layer
    z = np.matmul(a1,np.transpose(W2))    # (batch_size x 1)
    #print(np.shape(z))
    #print(z)
    p = sigmoid(z)
    #print(p)
    return [p,a1,h1]


# In[440]:


def backprop(W,p,a1,h1,X,y,loss):
    W1,W2 = W
    
    grad_loss = np.sum(y*(p-1)+(1-y)*p)            # d(Loss)/dP
    #print("Grad_loss: ",grad_loss)
    grad_z = sigmoid_grad(p)                       # dP/dZ
    #print("Grad_z: ",grad_z)
    #print(np.shape(grad_z),np.shape(a1),np.shape(grad_loss))
    #print("a1: ",a1)
    grad_W2 = np.matmul(np.transpose(grad_z),a1)*grad_loss     # dP/dW2 = (dP/dZ)*(dZ/dW2) ; dZ/dW2 = a1
    #print("grad_W2: ",grad_W2)
    #print(np.shape(X),np.shape(h1),np.shape(W2),np.shape(grad_z))
    #print(np.shape(np.multiply(grad_z,W2)))
    dp_dh1 = np.multiply(np.multiply(grad_z,W2),ReLU_grad(h1))    # dP/dh1
    #print("dP/dh1: ",np.sum(dp_dh1))
    #print("W2: ",W2)
    #print("Relu_grad: ",ReLU_grad(h1))
    grad_W1 = np.matmul(np.transpose(dp_dh1),X)*grad_loss
    
    # dP/dW1 = (dP/dZ)*(dZ/da1)*(da1/dh1)*(dh1/dW1)    
    #print("W1: "+str(np.sum(grad_W1)))
    #print("W2: "+str(np.sum(grad_W2)))
    # Update weights
    W1 = W1 - learning_rate*grad_W1
    W2 = W2 - learning_rate*grad_W2
    W = [W1,W2]
    return W


# In[441]:


def cross_entropy_loss(p,y,epsilon=1e-3):
    # p: batch_size X 1
    # y: batch_size X 1
    
    N = np.shape(p)[0]
    p = np.clip(p,epsilon,1.0-epsilon)
    #print("p: "+str(p)+"\ny: "+str(y))
    loss = -(np.sum(y*np.log(p) + (1-y)*np.log(1-p)))/N
    #print("Loss shape: "+str(np.shape(loss)))
    return loss


# In[442]:


def predict(X,W):
    p = forward(X,W)[0]
    y_pred = np.zeros(np.shape(p))
    for i in range(np.shape(p)[0]):
        if(p[i]>0.5):
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred


# In[455]:


def train(X,y,num_epochs=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    W = Weight_initialiser()
    costs = []
    j = 0
    for i in range(num_epochs):
        #print("i"+str(i))
        minibatches = random_mini_batches(X_train, y_train)
        for minibatch in minibatches:
            #print("j="+str(j))
            #j+=1
            (minibatch_X, minibatch_Y) = minibatch
            
            # Forward pass
            p,a1,h1 = forward(minibatch_X,W)
            
            # Cost calculation
            loss = cross_entropy_loss(p,minibatch_Y)
            #if(j%5==0):
            #print(loss)
            # Backward pass
#             print(p)
#             print("W")
#             print(np.shape(W))
            W = backprop(W,p,a1,h1,minibatch_X,minibatch_Y,loss)
            
        y_pred_test = predict(X_test,W)
        test_acc = accuracy_score(y_test,y_pred_test)

        y_pred_train = predict(X_train,W)
        train_acc = accuracy_score(y_train,y_pred_train)
        
        
        #print("Cost after epoch "+ str(i)+"= " + str(loss))
        costs.append(loss)
    print("Test Accuracy: "+str(test_acc))
    print("Train Accuracy: "+str(train_acc))
        #print("\n")
        
        
    #plt.plot(costs)
    #plt.show()
    
    return [W,costs]


# In[456]:


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
W,costs = train(X,y)


# In[ ]:




