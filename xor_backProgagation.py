
#Import libraries
"""

import numpy as np
import matplotlib.pyplot as plt

"""# Declare initial parameters"""

x=np.array([[0,0,1,1],[0,1,0,1]])
y=np.array([[0,1,1,0]]) #1x4
n_y=1
n_x=2
n_h=2
m=x.shape[1]
lr=0.1
#np.random.seed(2)
w1=np.random.rand(n_h,n_x)
w2=np.random.rand(n_y,n_h)
losses=[]
print(w1)
print(w2)
print(m)

"""# Define sigmoid function"""

def sigmoid(z):
  z=1/(1+np.exp(-z))
  return z

"""#forward propagation function"""

def forward_pass(w1,w2,x):
    z1=np.dot(w1,x)        #size 2x4
    a1=sigmoid(z1)         #2x4
    z2=np.dot(w2,a1)       #size 1x4
    a2=sigmoid(z2)         #1x4
    return z1,a1,z2,a2

"""# backward propagation function"""

def backward_pass(m,w1,w2,z1,a1,z2,a2,y):
    dz2=a2-y                              #1x4
    dw2=np.dot(dz2,a1.T)/m                #1x2
    dz1=np.dot(w2.T,dz2)*a1*(1-a1)        #2x2
    dw1=np.dot(dz1,x.T)/m                 #2x4
    #dw1=np.reshape(dw1,w1.shape)
    #dw2=np.reshape(dw2,w2.shape)
    return dz2,dw2,dz1,dw1

"""#number of iterations or epochs"""

iterations=10000

for i in range(iterations):
    z1,a1,z2,a2=forward_pass(w1,w2,x)
    loss=-(1/m)*np.sum(y*np.log(a2)+(1-y)*np.log(1-a2))
    losses.append(loss)
    dz2,dw2,dz1,dw1=backward_pass(m,w1,w2,z1,a1,z2,a2,y)
    w2=w2-lr*dw2                                          #1x2
    w1=w1-lr*dw1                                          #2x4

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss Values")

"""#Predict the model"""

def predict(w1,w2,input):
    z1,a1,z2,a2=forward_pass(w1,w2,test)
    if a2>=0.5:
        print("For input", [i[0] for i in input],"output is 1")
    else:
        print("For input", [i[0] for i in input],"output is 0")

test=np.array([[0.9],[0.2]])
predict(w1,w2,test)
test=np.array([[0],[0]])
predict(w1,w2,test)
test=np.array([[0],[1]])
predict(w1,w2,test)
test=np.array([[1],[1]])
predict(w1,w2,test)
