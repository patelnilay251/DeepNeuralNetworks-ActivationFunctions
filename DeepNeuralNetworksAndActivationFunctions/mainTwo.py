import numpy as np


class Parameters:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(1, output_size)

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
        self.layer = None  # Reference to parent layer

    def set_parameters(self, parameters):
        self.weights = parameters.get_weights()
        self.bias = parameters.get_bias()

    def neuron(self, inputs):
        if self.weights is None or self.bias is None:
            raise ValueError("Weights and bias must be set before forward pass")

        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Inputs shape ({inputs.shape}) is incompatible with weights shape ({self.weights.shape})"
            )

        self.aggregate_signal = np.dot(inputs, self.weights) + self.bias
        self.activation = self.layer.activation(self.aggregate_signal)
        self.output = self.activation


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


class Layer:
    def __init__(self, input_size, output_size, activation_type):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]
        self.parameters = Parameters(input_size, output_size)
        self.activation_type = activation_type
        self.activation = Activation(activation_type)
        self.neurons_layer = output_size

        # Set parameters for neurons
        for neuron in self.neurons:
            neuron.set_parameters(self.parameters)
            neuron.layer = self

    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            neuron.neuron(inputs)
            outputs.append(neuron.output)
        return np.array(outputs)


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def compute_loss(self, predicted, target):
        return np.mean(np.square(predicted - target))


# Example usage
input_size = 3
num_neurons = 3
num_layers = 3

inputs = np.random.randint(3, size=(3, input_size))
target = np.random.rand(3, num_neurons)

layers = [Layer(input_size, num_neurons, "sigmoid") for _ in range(num_layers)]
network = NeuralNetwork(layers)
outputs = network.forward(inputs)
loss = network.compute_loss(outputs, target)

print("Input:", inputs)
print("\nOutput:", outputs)
print("\nLoss:", loss)
