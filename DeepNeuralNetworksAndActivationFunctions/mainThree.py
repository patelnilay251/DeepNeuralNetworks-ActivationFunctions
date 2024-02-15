import numpy as np


class Parameters:
    def __init__(self):
        self.weights = None
        self.bias = None

    def set_bias(self, bias):
        self.bias = bias

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def set_weights(self, weights):
        self.weights = weights


class Neuron:
    def __init__(self, input_size):
        self.weights = None
        self.bias = None
        self.input_size = input_size
        self.aggregate_signal = None
        self.activation = None
        self.output = None
        self.grad_weights = None
        self.grad_bias = None

    def neuron(self, inputs):
        if self.weights is None or self.bias is None:
            raise ValueError("Weights and bias must be set before forward pass")

        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Inputs shape ({inputs.shape}) is incompatible with weights shape ({self.weights.shape})"
            )

        self.aggregate_signal = np.sum(np.dot(inputs, self.weights.T) + self.bias)
        self.activation = self.layer.activation(self.aggregate_signal)
        self.output = self.activation

    def compute_gradients(self, inputs, delta):
        self.grad_weights = np.dot(inputs.T, delta)
        self.grad_bias = np.sum(delta, axis=0)
        delta_next = np.dot(delta, self.weights)
        return delta_next


class Activation:
    def __init__(self, type):
        self.type = type

    def __call__(self, inputs):
        if self.type == "relu":
            return np.maximum(0, inputs)
        elif self.type == "sigmoid":
            return 1 / (1 + np.exp(-inputs))
        else:
            raise ValueError(f"Invalid activation function type: {self.type}")

    def derivative(self, inputs):
        if self.type == "relu":
            return np.where(inputs > 0, 1, 0)
        elif self.type == "sigmoid":
            return self(inputs) * (1 - self(inputs))
        else:
            raise ValueError(f"Invalid activation function type: {self.type}")


class Layer:
    def __init__(self, neurons, parameters, activation_type):
        self.neurons = neurons
        self.parameters = parameters
        self.weights = self.parameters.get_weights()
        self.bias = self.parameters.get_bias()
        self.activation_type = activation_type
        self.activation = Activation(activation_type)
        self.neurons_layer = len(neurons)

    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            neuron.weights = np.random.rand(self.neurons_layer)
            neuron.bias = self.bias
            neuron.layer = self
            neuron.neuron(inputs)
            outputs.append(neuron.output)
        return np.array(outputs)

    def backward(self, inputs, deltas):
        new_deltas = []
        for neuron, delta in zip(self.neurons, deltas):
            new_delta = neuron.compute_gradients(inputs, delta)
            new_deltas.append(new_delta)
        return np.array(new_deltas)


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def backward(self, inputs, deltas):
        new_deltas = deltas
        for layer in reversed(self.layers):
            new_deltas = layer.backward(inputs, new_deltas)
        return new_deltas

    def compute_loss(self, predicted, target):
        # For simplicity, let's use mean squared error as the loss function
        return np.mean(np.square(predicted - target))

    def update_parameters(self, learning_rate):
        for layer in self.layers:
            layer.parameters.weights -= learning_rate * layer.neurons[0].grad_weights
            layer.parameters.bias -= learning_rate * layer.neurons[0].grad_bias


# Example usage
input_size = 3
num_neurons = 3

inputs = np.random.randint(3, size=(input_size, num_neurons))
target = np.random.rand(input_size, num_neurons)  # Example target for demonstration

parameters = Parameters()
parameters.set_bias(np.random.rand(input_size))
parameters.set_weights(np.random.rand(num_neurons, input_size))

layer = Layer([Neuron(input_size) for _ in range(num_neurons)], parameters, "sigmoid")
outputs = layer.forward(inputs)

network = NeuralNetwork([layer])

# Forward pass
outputs = network.forward(inputs)

# Compute loss
loss = network.compute_loss(outputs, target)

# Backward pass
deltas = (
    2 * (outputs - target) / input_size
)  # Gradient of mean squared error loss function
deltas = network.backward(inputs, deltas)

# Update parameters
learning_rate = 0.01
network.update_parameters(learning_rate)

print("Input:", inputs)
print("\nOutput:", outputs)
print("\nLoss:", loss)
