#!/usr/bin/env python
# coding: utf-8

# In[37]:


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


# In[38]:


data = pd.read_csv('Assignment_4_data.txt', sep='\t',header=None).values


# In[39]:


x = data[:,1]
y = data[:,0]


# In[40]:


import nltk
nltk.download('stopwords')
stop_words = (set(stopwords.words("english")))


# In[41]:


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


# In[42]:


# print(len(sentences))
# print(len(vocab))
num_sentences = len(sentences)
vocab_size = len(set(vocab))
print(vocab_size)
print(num_sentences)
#print(set(vocab))

vocab = list(set(vocab))
#print(vocab)


# In[43]:


word_2_idx = {}
i = 0
for word in vocab:
    word_2_idx[word] = i
    i+=1


# In[44]:


# One-hot vector generation for all sentences
X = np.zeros((num_sentences,vocab_size))
for i in range(0,len(sentences)):
    for j in range(0,len(sentences[i])):
        word = sentences[i][j]
        idx = word_2_idx[word]
        X[i][idx] = 1


# In[45]:


le = LabelEncoder()
le.fit(y)
y = le.transform(y)


# In[46]:


def sigmoid(x):
    #print(np.shape(x))
    sig = np.exp(x)/(1 + np.exp(x))
    #print(np.shape(sig))
    return sig


# In[47]:


def sigmoid_grad(sig):
    '''Takes value of sigmoid as input and returns gradient w.r.t x '''
    # x is not taken as an input
    return sig*(1-sig)


# In[48]:


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# In[58]:


# Dimensions of Neural Network

input_dim = vocab_size  # input dimension
n1 = int(input("Enter the number of neurons in 1st hidden layer: "))
n2 = int(input("Enter the number of neurons in 2nd hidden layer: "))

output_dim = 1   # output dimension

# Other hyperparameters 
learning_rate = 0.1


# In[59]:


def Weight_initialiser():
    W1 = np.random.uniform(size=(n1,input_dim),low=-1,high=1)    # n1*7375
    W2 = np.random.uniform(size=(n2,n1),low=-1,high=1)           # n2*n1
    W3 = np.random.uniform(size=(output_dim,n2),low=-1,high=1)   # 1*n2
#     W1 = np.random.normal(size=(hidden_1_dim,input_dim))    
#     W2 = np.random.normal(size=(output_dim,hidden_1_dim))   
    return [W1,W2,W3]


# In[60]:


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
#     if m % mini_batch_size != 0:
#         mini_batch_X = shuffled_X[num_complete_minibatches*mini_batch_size:m,:]
#         mini_batch_Y = shuffled_Y[num_complete_minibatches*mini_batch_size:m]
        
#         mini_batch = (mini_batch_X, mini_batch_Y)
#         mini_batches.append(mini_batch)
    
    return mini_batches


# In[61]:


def forward(X,W):
    # X: input matrix (batch_size x 7375)
    W1,W2,W3 = W
    # W1: (n1 x 7375)
    # W2: (n2 x n1)
    # W3: (1 x n2)
    #print(np.shape(W1),np.shape(W2))
    
    # First hidden layer and activation layer
    h1 = np.matmul(X,np.transpose(W1))    # (batch_size x 100)
    a1 = sigmoid(h1)
    
    # Second hidden layer
    h2 = np.matmul(a1,np.transpose(W2))
    a2 = sigmoid(h2)
    # Output layer
    z = np.matmul(a2,np.transpose(W3))    # (batch_size x 1)
    #print(np.shape(z))
    #print(z)
    p = softmax(z)
    #print(p)
    return [p,a1,h1,a2,h2]


# In[62]:


def backprop(W,p,a1,h1,a2,h2,X,y,loss):
    W1,W2,W3 = W
    
    grad_softmax_score = p - y
    
    
    grad_W3 = np.matmul(h2.T,grad_softmax_score)
    grad_h2 = np.matmul(grad_softmax_score,W3.T)
    grad_a2 = (grad_relu(a2)*grad_h2)
    grad_W2 = np.matmul(grad_a2,h1.T)
    grad_h1 = np.matmul(W2.T,grad_a2)
    grad_a1 = (grad_relu(a1)*grad_h1)
    grad_W1 = np.matmul(grad_a1,x_batchinput)
    
    W3 -= learning_rate * grad_W3
    W2 -= learning_rate * grad_W2
    W1 -= learning_rate * grad_W1
    
    W = [W1,W2,W3]
    return W


# In[63]:


def cross_entropy_loss(p,y,epsilon=1e-3):
    # p: batch_size X 1
    # y: batch_size X 1
    
    N = np.shape(p)[0]
    p = np.clip(p,epsilon,1.0-epsilon)
    #print("p: "+str(p)+"\ny: "+str(y))
    loss = -(np.sum(y*np.log(p) + (1-y)*np.log(1-p)))/N
    #print("Loss shape: "+str(np.shape(loss)))
    return loss


# In[64]:


def predict(X,W):
    p = forward(X,W)[0]
    y_pred = np.zeros(np.shape(p))
    for i in range(np.shape(p)[0]):
        if(p[i]>0.5):
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred


# In[65]:


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
            p,a1,h1,a2,h2 = forward(minibatch_X,W)
            
            # Cost calculation
            loss = cross_entropy_loss(p,minibatch_Y)
            #if(j%5==0):
            #print(loss)
            # Backward pass
#             print(p)
#             print("W")
#             print(np.shape(W))
            W = backprop(W,p,a1,h1,a2,h2,minibatch_X,minibatch_Y,loss)
            
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


W,costs = train(X,y)




