import numpy as np
from floodData import *
from split import data_split
class MLP(object):
    output_l = []
    def __init__(self, num_inputs = 8, hidden_layers = [7], num_outputs = 1):
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        layers = [num_inputs] + hidden_layers + [num_outputs]
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
    def forward_propagate(self, inputs):
        activations = inputs
        self.activations[0] = activations
        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations, w)
            activations = self._sigmoid(net_inputs)
            self.activations[i + 1] = activations
        return activations
    def back_propagate(self, error):
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            delta_re = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations = current_activations.reshape(current_activations.shape[0],-1)
            self.derivatives[i] = np.dot(current_activations, delta_re)
            error = np.dot(delta, self.weights[i].T)
    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_errors = 0
            for j, input in enumerate(inputs):
                target = targets[j]
                output = self.forward_propagate(input)
                error = target - output
                self.back_propagate(error)
                self.gradient_descent(learning_rate)
                sum_errors += self._mse(target, output)
            self.output_l.append(output)
            print("Error: {:.20f} at epoch {}".format(sum_errors / len(items), i+1))
        print("train complete laew jaa eiei!!!")
        print("-----------------------------------------------")
    def gradient_descent(self, rate = 1):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * rate
    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y
    def ReLU(self,Z):
        return np.maximum(Z, 0)
    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)
    def ReLU_deriv(Z):
        return Z > 0
    def _mse(self, target, output):
        return np.average((target - output) ** 2)

error = []
desired = []
calculate = []
k=10
folds = data_split(f_d, k)
for i in range(k):
    _k = len(folds[i])
    items = np.array([folds[i][j][0:len(folds[i][j])-1] for j in range(len(folds[i]))])
    targets = np.array([folds[i][j][len(folds[i][j])-1] for j in range(len(folds[i]))])
    mlp = MLP(8, [5], 1)
    mlp.train(items, targets, 1000, 0.6)
    _input = folds[i][_k-1][0:len(folds[i][_k-1])-1]
    input = np.array(_input)
    target = np.array((folds[i][_k-1][len(folds[i][_k-1])-1]))
    output = mlp.forward_propagate(input)
    desired.append(denor(target))
    calculate.append(denor(output))
    print()
    print("Station 1's water level = {} \nStation 2's water level = {} \nWater level in 7 more hours = {}".format(denor(input[0:4]),
                                                                                                                  denor(input[4:8]), denor(output)))
    print("but it should be {}".format(denor(target)))
    print()
    error.append(abs(denor(target)-denor(output))*100/denor(target))
for i in range(10):
    print("error round {} : {:2.5f} %".format(i,error[i][0]))
print()
