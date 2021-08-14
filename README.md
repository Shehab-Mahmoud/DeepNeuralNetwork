# Deep Neural Network
Python implementation of deep neural network  from scratch with a mathmatical approach.

## Table of contents:
1. Overview
2. Intializing paramaters
3. Forward propagation
4. [Cost function](https://github.com/Shehab-Mahmoud/DeepNeuralNetwork#2cost-function)
5. Backward propagation
6. Training (gradient descent)
5. predict

## 1.Forward propagation :
Forward propagation is mainly broken into two steps:
1. linear forward (weighted sum input):<br>
    &ensp;calculating z = w.x + b<br><br>

2. activation:<br>
   &ensp;pluging z into the activation function sigmoid or relu ...<br>
    &ensp;A = g(z)<br>

<p align='center'>
<img src="Images/forward-prop2.png" width="500" height="200"> 
</p>

For an L-layer model we commonly use the [relu activation function](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) for hidden layers neurons<br>
and [sigmoid activation function](https://en.wikipedia.org/wiki/Activation_function) for output layer in case of binary classification<br>
as it maps values to propablties between 0 and 1<br>
therefor: <br>
propabilty \> 0.5 = 1 , and propabilty \< 0.5 = 0<br><br>

In case of multi-class classification we use a [Softmax activation function](https://en.wikipedia.org/wiki/Softmax_function) in output layer<br>
which isn't implemented here and i will implement later.<br>

<p align='center'>
<img src="Images/forward-prop-n.jpg" width="500" height="300"> 
</p>

Here is a vectorized implementation of forward propagation:<br>

<p align='center'>
<img src="Images/vectorized-forward.png" width="300" height="300"> 
</p>

## 2.Cost Function:
Since we are doing binary classification we use the logistic cost function

<p align='center'>
<img src="Images/logistic-cost.png" width="700" height="200"> 
</p>

## 3.Back propagation
Back propagation is the step that allows calculating gradients for gradient descent (training the neural network).<br>
In back propagation we follow the reversed path of the neural network calculating gradients for weights and biases to update them
during gradient descent (training).<br><br>
This is a very usefull [article](https://medium.com/@pdquant/all-the-backpropagation-derivatives-d5275f727f60) explaining the math behind back propagation, essentialy we use the chain rule from calculas to calculate the derivative of loss w.r.t weights and biases.

<p align='center'>
<img src="Images/back-prop.png" width="700" height="300"> 
</p>

The reason we use the chain rule maybe not be very obvious for some people so we can break it down this way: <br>
* let the cost function be: **L(g)**
* let the activation function be: **g = g(z)**
* let the weighted sum be: **z = z(w,b)**<br>

So the loss is **L(g(z(w,b)))**<br>
How to get the ***gradient*** of this function w.r.t ***w,b*** ? <br>
> we can do this using the chain rule from calculus

we simply break down the equation into partial derivatives of loss w.r.t w,b<br>

<p align='center'>
<img src="Images/back-eq.PNG" width="500" height="100"> 
</p>

### The steps to implementing back propagation :
1. kick start back prop. by calculating the drivative of Loss w.r.t last layer activation
2. since the last layer(output layer) is unique (has diffrent activation from other layers), calculate the derivative of the loss
w.r.t weights and biases
> Here we use the sigmoid activation function so we use its derivative.

3. loop over all of the rest of the layers calculating gradients and storing them.
> Here we use the relu activation function so we use its derivative.

Here is the vectorized implementation of back propagation:

<p align='center'>
<img src="Images/vec-back.png" width="300" height="400"> 
</p>


## Links :
* [loading data](https://github.com/Shehab-Mahmoud/DeepNeuralNetwork/blob/main/load_data.py)
* [neural network class](https://github.com/Shehab-Mahmoud/DeepNeuralNetwork/blob/main/DNN.py)