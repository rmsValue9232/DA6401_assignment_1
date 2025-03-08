import numpy as np
import numpy.typing as npt
from activations import Activation

class Layer:
    """
    A single layer in a neural network.
    - Here, a layer refers to the collection of the weights, biases and activation function.
    - It will be responsible for finding activated output from provided input to the layer.
    - Layer instantiation require the index of the layer, its input size and output size.
    - Optionally can also provide weight initialization method and also the activation function to use.

    Attributes
    ----------
    `index`: `int`, >= 1
        The index of the layer when used in a NN.
    `input_size`: `int`, >= 1
        The incoming input size that needs to activated.
    `output_size`: `int`, >= 1
        The activated output formed from the forward pass through the layer.
    `weight_init`: `str`, ["random", "xavier"], default = "random"
        Layer's weights initialization method to use.
    `activation`: `str`, ["identity", "sigmoid", "tanh", "relu", "softmax"],  default = "identity"
        Specify which activation function to use in forward pass
    
    Methods
    -------
    `forward(x: NDArray)`: `NDArray`
        Takes the input and passes through this layer to give activated output


    """
    _weight_init_methods = ["random", "xavier"]
    _activation_choices = ["identity", "sigmoid", "tanh", "relu", "softmax"]

    def __init__(self, index: int, input_size: int, output_size: int, weight_init = "random", activation="identity"):

        assert (index >= 1), "Layer index should be greater than or equal to 1"
        assert (input_size >= 1), "Input side of the layer needs to have atleast one neuront"
        assert (output_size >= 1), "Output side of the layer needs to have atleast one neuron"

        
        if weight_init not in self._weight_init_methods:
            raise ValueError(f"Unknown weight initialization method '{weight_init}', choose from: {self._weight_init_methods}")
        
        if activation not in self._activation_choices:
            raise ValueError(f"Unknown activation choice '{activation}', choose from: {self._activation_choices}")

        self.index = index
        self.weight_init = weight_init
        self.input_size = input_size
        self.output_size = output_size
        self.activation = Activation(activation)
        self.W, self.b = self._init_w_b(input_size, output_size)
    
    def _init_w_b(self, input_size: int, output_size: int):
        if self.weight_init == "random":
            W = np.random.randn(output_size, input_size) * 0.01
            b = np.zeros((output_size, 1))
            return (W, b)
        elif self.weight_init == "xavier":
            b = np.zeros((output_size, 1))
            n = input_size
            m = output_size
            
            # based on https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
            if (self.activation.choice == "relu"):
                sd = np.sqrt(2/n) # standard deviation of zero-mean normal to use for sampling weights
                W = np.random.normal(loc=0.0,
                                     scale=sd,
                                     size=(m, n))
            else:
                low = -np.sqrt(6)/np.sqrt(n+m)
                high = -low # specify the range of uniform distribution to use for sampling weights
                W = np.random.uniform(low=low,
                                      high=high,
                                      size=(m, n))
            
            return (W, b)
    
    def forward(self, x: npt.NDArray) -> npt.NDArray:
        self.a = (self.W @ x) + self.b
        self.h = self.activation.function(self.a)
        return self.h

