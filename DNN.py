import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


class activate:
    z = 0
    A = 0
    dA = 0
    def __init__(self,z):
        '''
        construct the class with the values of z to preform operations
        like calculate activation or backward activation 
        '''
        self.z = z
    
    def sigmoid(self):
        '''
        calculates the sigmoid of z
        and stores the result in self.A
        '''
        # using scipy.special.expit instead of 1/1+np.exp(-z) to avoid overflow
        self.A = expit(self.z)
        assert self.z.shape == self.A.shape, 'incorrect shape of A for sigmoid (A != Z) in sigmoid'

    def relu(self):
        '''
        calculates the relu of z
        and stores the result in self.A
        '''
        self.A = np.maximum(0,self.z)
        assert self.A.shape == self.z.shape , 'incorrect shape of A for relu (A != Z) in relu'

    def sigmoid_back(self):
        '''
        calculates the derivative of sigmoid
        stores the result in self.dA
        '''
        self.sigmoid()
        self.dA = self.A*(1-self.A)
        assert self.z.shape == self.dA.shape, 'incorrect shape of dA for sigmoid (dA != Z) in sigmoid_back'

    def relu_back(self):
        '''
        calculates the derivative of relu 
        stores the result in self.dA
        '''
        self.z[self.z<=0] = 0
        self.z[self.z>0]= 1
        self.dA = self.z

class DNN:
    '''
    This class is an implementation form scratch for deep neural network
    it is not yet developed to be generic to any any type of data.

    methods:
        -calculate_z
        -forward_prop
        -calculate_cost
        -back_prop
        -update_params
        -train_model ---> used by user
        -predict ----> used by user
    '''
    params = {}
    Ldims =0
    z = {}
    a = {}
    m = 0
    X_train = 0
    y_train = 0
    costs = []
    iters = []

    grads = {}

    def __init__(self,Ldims,X_train,y_train):
        '''
        initialize paramaters of L layer model where :
        W shape = n[l],n[l-1]
        b shape = n[l],1
        w : values intialied at random using he/xavier intialization to prevent exploding gradients
            as it keeps the variance between weights at 2/n keeping w a bit above and below 0
            1/np.sqrt(2/n[l-1]) where n : number of neurons in previous layer
            we can tune this but we wont do that.
        b : zeros

        -w,b are stored in the params dictionary

        -stores the normalized training set as self.X_train and self.y_train
        -stores the number of training samples as self.m
        -stores the layers dimentions in the self.Ldims
        '''
        self.Ldims = Ldims
        np.random.seed(42)
        for l in range(1,len(Ldims)):
            self.params['W'+str(l)] = np.random.randn(Ldims[l],Ldims[l-1])*np.sqrt(2./Ldims[l-1])
            self.params['b'+str(l)] = np.zeros((Ldims[l],1))

            assert self.params['W'+str(l)].shape == (Ldims[l],Ldims[l-1]),'incorrect shape of intialized weights'
            assert self.params['b'+str(l)].shape == (Ldims[l],1) , 'incorrect shape of intialized biases'

        self.X_train = X_train
        self.y_train = y_train
        self.m = X_train.shape[1]

    @staticmethod
    def calculate_z(w,a_prev,b):
        '''
        calculates the linear activation z = w.a + b
        parameters:
            w : weights of the layer stored in params dictionary
            b : biases of the layer stored in params dictionary
            a_prev : activation of the previous layer
        
        returns:
            z
        '''
        z = w.dot(a_prev)+b
        return z 

    def forward_prop(self):
        '''
        implements forward propagation of the neural network training
        by:
        1- setting intial activations as the input features
        2- loop over each layer:
                1- setting intial activations as the input features
                2- calculate z = w.A + b
                3- calculate activation a=relu(z)
        3- calculate activation for last layer as sigmoid(z)

        values of z are stored in self.z
        values of a are stored in self.a
        '''

        a_prev = self.X_train
        
        for l in range(1,len(self.Ldims)-1):
            
            self.z['z'+str(l)] = self.calculate_z(self.params['W'+str(l)],a_prev,self.params['b'+str(l)])
            assert self.z['z'+str(l)].shape == (self.Ldims[l],self.m) , 'incorrect shape of z in forward prop'

            activation = activate(self.z['z'+str(l)])
            activation.relu()
            self.a['A'+str(l)] = activation.A
            assert self.a['A'+str(l)].shape == (self.Ldims[l],self.m), 'incorrect shape of A after using relu in forward prop'

            a_prev = self.a['A'+str(l)]

        # applying sigmoid for last layer    
        L = len(self.Ldims)-1
        self.z['z'+str(L)] = self.calculate_z(self.params['W'+str(L)],a_prev,self.params['b'+str(L)])
        activation = activate(self.z['z'+str(L)])
        activation.sigmoid()
        self.a['A'+str(L)] = activation.A



    def calculate_cost(self):
        '''
        calculates the cost of the network output
        
        returns :
            cost
        '''
        # number of layers
        L = len(self.Ldims)-1
        # training true output
        y = self.y_train
        # training output
        A = self.a['A'+str(L)]
        # computing cost
        cost = (1./self.m) * (-np.dot(y,np.log(A).T) - np.dot(1-y, np.log(1-A).T))
        return cost.reshape(1)        
        

    def back_prop(self):
        '''
        preforms back propagation for neural network training.
        1- intialize the activations for layer zero as the input features

        2- kick start the back prop by:
            1- calculating dL/dA (y/a)+(1-y)/(1-a)
            2- calculate dz=dz/dL which the dA/dz * dL/dA (using the derivative of sigmoid)
            3- calculate dw and db

        3- start the rest of the back prop loop :
            1- calculate relu_backward[l]
            2- calculate dz[l]
            3- calculate dw,db
            4- calculate dA[l-1] 
        
        all gradient values are stored in the grads dictionary
        '''
        self.a['A0'] = self.X_train
        L = len(self.Ldims)-1
        # intialize back propagation
        self.grads['dA'+str(L)] = -self.y_train/self.a['A'+str(L)] +(1-self.y_train)/(1-self.a['A'+str(L)])

        activation = activate(self.z['z'+str(L)])
        activation.sigmoid_back()
        self.grads['dz'+str(L)] = self.grads['dA'+str(L)]*activation.dA

        self.grads['dW'+str(L)] = (1./self.m)*np.dot(self.grads['dz'+str(L)],self.a['A'+str(L-1)].T)
        self.grads['db'+str(L)] = (1./self.m)*np.sum(self.grads['dz'+str(L)],axis=1,keepdims=True)
        assert self.grads['db'+str(L)].shape == self.params['b'+str(L)].shape ,'shapes of db != b'
        assert self.grads['dW'+str(L)].shape == self.params['W'+str(L)].shape ,'shapes of dW != W'

        self.grads['dA'+str(L-1)] = np.dot(self.params['W'+str(L)].T,self.grads['dz'+str(L)])
        assert self.grads['dA'+str(L-1)].shape == self.a['A'+str(L-1)].shape ,'shapes of dA != A'
    

        for l in reversed(range(L-1)):
            
            activation = activate(self.z['z'+str(l+1)])
            activation.relu_back()
            self.grads['dz'+str(l+1)] = self.grads['dA'+str(l+1)]*activation.dA

            self.grads['dW'+str(l+1)] = (1./self.m)*np.dot(self.grads['dz'+str(l+1)],self.a['A'+str(l)].T)
            self.grads['db'+str(l+1)] = (1./self.m)*np.sum(self.grads['dz'+str(l+1)],axis=1,keepdims=True)
            assert self.grads['db'+str(l+1)].shape == self.params['b'+str(l+1)].shape ,'shapes of db != b'
            assert self.grads['dW'+str(l+1)].shape == self.params['W'+str(l+1)].shape ,'shapes of dW != W'

            self.grads['dA'+str(l)] = np.dot(self.params['W'+str(l+1)].T,self.grads['dz'+str(l+1)])
            assert self.grads['dA'+str(l)].shape == self.a['A'+str(l)].shape ,'shapes of dA != A'

    def update_params(self,lr):
        '''
        updates the parameters w,b during training of the neural netowrk

        paramters :
            lr : learning rate
        '''
        L = len(self.Ldims)-1
        for i in range(L):
            self.params['W'+str(i+1)] = self.params['W'+str(i+1)] - lr*self.grads['dW'+str(i+1)]
            self.params['b'+str(i+1)] = self.params['b'+str(i+1)] - lr*self.grads['db'+str(i+1)]


    def train_network(self,lr,n_iters,print_cost = False,plot_cost = False):
        '''
        train the network with the predefined parameters

        parameters :
            lr : learning rate
            n_iters : number of iterations of training
            print_cost : True -> print the costs every 100 iterations
            plot_cost  : plots the costs vs iterations
        '''
        # 1- forward prop
        for i in range(n_iters):
            self.forward_prop()
            # 2- compute cost
            cost = self.calculate_cost()            
            # 3- backprop
            self.back_prop()
            # 4- update params
            self.update_params(lr)

            # printing costs
            if print_cost and i % 100 == 0 or i == n_iters - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == n_iters:
                self.costs.append(cost)
                self.iters.append(np.array([[i]]).reshape(1))

            #plotting costs
        if plot_cost==True:
            plt.plot(self.iters,self.costs)
            plt.show()
        
            
    def predict(self,sample):
        '''
        preforms predictions on the trained network
        and retunrs predicted probablties > 0.5 as 1
        and predicted probablties < 0.5 as 0

        paramters :
            sample : values of which you want to preform prediction on
        '''
        a_prev = sample
        
        for l in range(1,len(self.Ldims)-1):
            
            z = self.calculate_z(self.params['W'+str(l)],a_prev,self.params['b'+str(l)])
            
            activation = activate(z)
            activation.relu()
            a = activation.A

            a_prev = a

        # applying sigmoid for last layer    
        L = len(self.Ldims)-1
        z = self.calculate_z(self.params['W'+str(L)],a_prev,self.params['b'+str(L)])
        z = np.array(z,dtype=np.longdouble)
        
        activation = activate(z)
        activation.sigmoid()
        propas = activation.A

        # if sigmoid output > 0.5 therefor 1 else its 0
        for i in range(propas.shape[1]):
            if propas[0,i] >0.5:
                propas[0,i] =1
            else:
                propas[0,i] = 0
      

        return propas

    @staticmethod
    def accuracy(y_true,y_pred):
        '''
        calculates accuracy of true and predicted values
        parameters :
            y_true: true values
            y_pred: predicted values

        returns:
            accuracy
        '''
        m = y_pred.shape[1]
        accuracy = np.sum((y_pred == y_true)/m)
        return accuracy
            
     

