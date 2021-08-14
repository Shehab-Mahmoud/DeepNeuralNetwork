# Deep Neural Network
Python implementation of deep neural network  from scratch with a mathmatical approach.

## Table of contents:
1. Overview
2. Intializing paramaters
3. Forward propagation
4. Backward propagation
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

## Links :
* [loading data](https://github.com/Shehab-Mahmoud/DeepNeuralNetwork/blob/main/load_data.py)
* [neural network class](https://github.com/Shehab-Mahmoud/DeepNeuralNetwork/blob/main/DNN.py)