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
    &ensp;A = g(z)<br><br>

<p align='center'>
<img src="Images/forward-prop2.png" width="500" height="200"> 
</p>

For an L-layer model we commonly use the **relu** activation function for hidden layers neurons<br>
and **sigmoid** activation function for output layer in case of binary classification<br><br>
In case of multi-class classification we use a **Softmax** activation function<br>
which isn't implemented here and i will implement later.<br>

## Links :
* [loading data](https://github.com/Shehab-Mahmoud/DeepNeuralNetwork/blob/main/load_data.py)
* [neural network class](https://github.com/Shehab-Mahmoud/DeepNeuralNetwork/blob/main/DNN.py)