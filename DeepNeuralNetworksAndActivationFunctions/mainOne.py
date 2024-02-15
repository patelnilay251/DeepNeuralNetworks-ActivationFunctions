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


class Neuron:
    def __init__(self, input_size):
        self.weights = None
        self.bias = None
        self.input_size = input_size
        self.aggregate_signal = None
        self.activation = None
        self.output = None

    def neuron(self, inputs):
        # Check if weights and bias are initialized
        if self.weights is None or self.bias is None:
            raise ValueError("Weights and bias must be set before forward pass")

        # Check if input shape is compatible with weights shape
        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Inputs shape ({inputs.shape}) is incompatible with weights shape ({self.weights.shape})"
            )

        # Calculate the aggregate signal
        self.aggregate_signal = np.sum(np.dot(inputs, self.weights.T) + self.bias)
        # Apply activation function
        self.activation = self.layer.activation(self.aggregate_signal)
        # Set neuron output
        self.output = self.activation


class Activation:
    def __init__(self, type):
        self.type = type

    def __call__(self, inputs):
        # Apply the specified activation function
        if self.type == "linear":
            return inputs
        elif self.type == "relu":
            return np.maximum(0, inputs)
        elif self.type == "sigmoid":
            return 1 / (1 + np.exp(-inputs))
        elif self.type == "tanh":
            return np.tanh(inputs)
        elif self.type == "softmax":
            exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
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
        # Iterate through neurons in the layer
        for neuron in self.neurons:
            # Initialize weights for the neuron
            neuron.weights = np.random.rand(self.neurons_layer)
            neuron.bias = self.bias
            # Set the layer for the neuron
            neuron.layer = self
            # Compute output for the neuron
            neuron.neuron(inputs)
            outputs.append(neuron.output)
        return np.array(outputs)


# Input size and number of neurons
input_size = 2
num_neurons = 3

# Generate random inputs
inputs = np.random.randint(2, size=(input_size, num_neurons))

# Initialize parameters (bias)
parameters = Parameters()
parameters.set_bias(np.random.rand(input_size))

# Create a layer with neurons and specify activation function
layer = Layer([Neuron(input_size) for _ in range(num_neurons)], parameters, "sigmoid")

# Forward pass
outputs = layer.forward(inputs)

# Print input and output
print("Input:", inputs)
print("\nOutput:", outputs)
