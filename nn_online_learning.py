
# coding: utf-8

# In[14]:


import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


# In[15]:


# input array
#X = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
in_data_from_file = genfromtxt('cross_data.csv', delimiter=',', dtype=float)
X = in_data_from_file[...,:2]
Y = in_data_from_file[...,2:]

# output
#Y = np.array([[1],[1],[0]])

#print("X", X.shape)
#print("Y", Y.shape)


# In[16]:


# sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# derivative of sigmoid function
def derivatives_sigmoid(x):
    return x*(1-x)


# In[17]:


# variable initialization
inputlayer_neurons = X.shape[1] # number of feature in data set
hiddenlayer_neurons = 3 # number of hidden layer neurons
output_neurons = 1 # number of neurons at output layer


# In[18]:


# weight and bias information
#wh = np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
wh = genfromtxt('w1.csv', delimiter=',', dtype=float).T
#bh = np.random.uniform(size=(1,hiddenlayer_neurons))
bh = np.array([genfromtxt('b1.csv', delimiter=',', dtype=float)])
#wout = np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
#wout = np.array([[0.0511],[0.1611],[0.0238],[-0.0267],[0.1089],[0.2381],[0.0784],[0.003],[0.1646],[-0.1779]])
wout = np.array([genfromtxt('w2.csv', delimiter=',', dtype=float)]).T
#bout = np.random.uniform(size=(1,output_neurons))
bout = np.array([[genfromtxt('b2.csv', delimiter=',', dtype=float)]])

print ("wh", wh.shape)
print ("bh", bh.shape)
print ("wout", wout.shape)
print ("bout", bout.shape)


# In[19]:


lr = 0.7 # setting learning rate
mt = 0.3

i = 0
Err_avg = 1
Err_avg_per_epoch = np.array([])
epochs = np.array([])

while (Err_avg > 0.001):
    
    Err_sum = 0
    wout_previous = np.zeros_like(wout)
    bout_previous = np.zeros_like(bout)
    wh_previous = np.zeros_like(wh)
    bh_previous = np.zeros_like(bh)
    
    for x,y in zip(X,Y):
        #Forward Propogation
        x = np.array([x])
        hidden_layer_input1 = np.dot(x,wh)
        hidden_layer_input = hidden_layer_input1 + bh
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1 = np.dot(hiddenlayer_activations,wout)
        output_layer_input = output_layer_input1 + bout
        output = sigmoid(output_layer_input)

        #Backpropogation
        E = y - output
        slope_output_layer = derivatives_sigmoid(output)
        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
        d_output = E * slope_output_layer

        Error_at_hidden_layer = d_output.dot(wout.T)
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

        wout += ((hiddenlayer_activations.T.dot(d_output) * lr) + (mt*wout_previous))
        bout += ((np.sum(d_output, axis=0, keepdims=True) * lr) + (mt*bout_previous))

        wh += ((x.T.dot(d_hiddenlayer) * lr) + (mt*wh_previous))
        bh += ((np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr) + (mt*bh_previous))
        
        wout_previous = wout
        bout_previous = bout
        wh_previous = wh
        bh_previous = bh
        
        Err = (E*E)/2
        Err_sum += Err
    
    i += 1
    Err_avg = Err_sum/(len(X))
    print(i, Err_avg)
    Err_avg_per_epoch = np.append(Err_avg_per_epoch, Err_avg)
    epochs = np.append(epochs, i)
    Err_avg = 0.0001
    if i == 1:
        print("******************************************")
        print(wh)
        print("******************************************")
        print(bh)
        print("******************************************")
        print(wout)
        print("******************************************")
        print(bout)

print("******************************************")
print(wh)
print("******************************************")
print(bh)
print("******************************************")
print(wout)
print("******************************************")
print(bout)


# In[ ]:


print(i)
#training_fig = plt.figure()
#ax = training_fig.subplot(111)
#plt.plot(epochs, Err_avg_per_epoch)
#plt.xlabel("epochs")
#plt.ylabel("Average error energy")
#plt.show()
x = X[:,0:1]
y = X[:,1:2]
x = np.reshape(x, 314,)
y = np.reshape(y, 314,)

x_min, x_max = x.min() - 1, x.max() + 1
y_min, y_max = y.min() - 1, y.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 200),
                     np.arange(y_min, y_max, 200))

# here "model" is your model's prediction (classification) function
plotData = np.c_[xx.ravel(), yy.ravel()]
def predict(x):
    hidden_layer_input1 = np.dot(x,wh)
    hidden_layer_input = hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1 = np.dot(hiddenlayer_activations,wout)
    output_layer_input = output_layer_input1 + bout
    output = sigmoid(output_layer_input)
    return output
        
Z = predict(plotData)
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z)
#plt.axis('off')


plt.scatter(x,y,c=Y,cmap=plt.cm.Paired)
plt.show()


# In[ ]:




