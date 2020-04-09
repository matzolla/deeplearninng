import numpy as np
#hey ! today we are going to create a perceptron algorithm it is a simple linear line seperating to classes to the
#best of its kind...
#our main objectif is to create a linear boundary seperating two classes the line is of the form w_1*x_1 + w_2*x_2+...+ b =0
# when training the model we qre going to update the different the weights (w_1,w_2....) and the bias to
#obtain the best line
#we start by defining a simple function

def result(t):
    #this are the values assigned to y_hat after the inputs are being passed
    #into the prediction function ("1" if the value is greater than 0
    #and '0' if not)
    if t=> 0:
        return 1
    return 0

def predictions(W,X,b):
    #the weight is an array of numbers and the X is our different inputs
    return result((np.matmul(X,W)+b)[0])
def perceptron(W,X,y,learning_rate=0.01,b):
    # the x values appear as a list of tuple
    #if
    for i in range(len(X)):
        y_hat=predictions(W,X[i],b)
        if y[i]-y_hat==1:
            W[0] +=0.01*W[0]
            W[1] +=0.01*W[1]
            b += 0.01
        elif y[i]-y_hat==-1:
            W[0] -= 0.01*W[0]
            W[1] -= 0.01*W[1]
            b -= 0.01
        return W,b
def training(X,y,learning_rate=0.01,num_epoch=25):
    W=np.array(np.random.rand(2,1))
    b=np.random(1)[0]
    for i in range(num_epoch):
        W,b=perceptron(W,X,y,learning_rate=0.01,b)
    print('this are the weighte {} and the bias {}'.format(W,b))
