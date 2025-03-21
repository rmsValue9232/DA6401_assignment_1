import numpy as np
import numpy.typing as npt
from activations import Activation
from optimizers import Optimizer
from collections import deque
import losses
from utils import confusion_labels, accuracy

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
    """
    _weight_init_methods = ["random", "xavier"]

    def __init__(self, index: int, input_size: int, output_size: int, weight_init = "random", activation = "identity"):

        assert (index >= 1), "Layer index should be greater than or equal to 1"
        assert (input_size >= 1), "Input side of the layer needs to have atleast one neuront"
        assert (output_size >= 1), "Output side of the layer needs to have atleast one neuron"

        
        if weight_init not in self._weight_init_methods:
            raise ValueError(f"Unknown weight initialization method '{weight_init}', choose from: {self._weight_init_methods}")

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
        """
        Takes the input `x` and passes through this layer to give activated output `h`.

        Parameters
        ----------
        `x`: `NDArray`
            The input vector to feed to the layer.
        
        Returns
        -------
        `NDArray`
            The activated output of the layer after forward pass of `x`.
        """
        self.x = x
        self.a = (self.W @ x) + self.b
        self.h = self.activation.function(self.a)
        return self.h

class NeuralNetwork:
    """
    Implements multi-layered neural network with flat layers.

    - Can specify all the layers in one go.
    - Or iteratively add new layers in front of the last layer.

    Attributes
    ----------
    `weight_init`: `str`, ["random", "xavier"], default = "random"
        All layers' weights initialization method to use.
    `layer_sizes`: `list`, default = None
        A list of strictly positive integers specifying number of neurons in each layer of the network. Can leave this empty if intend to use
    `activations`: `list`, default = None
        A list of activation function name strings to use for all the activation layers (hidden + output)
    """
    def __init__(self, weight_init = "random", layer_sizes: list = None, activations:list = None, optimizer = "sgd", learning_rate = 0.001, weight_decay = 0.0, **kwargs):
        self.opt_name = optimizer
        self.lr = learning_rate
        self.wd = weight_decay
        self.optimizer_params = kwargs
        self.weight_init = weight_init
        self.layers = []
        
        if layer_sizes is None:
            print("Neural Network initialized without layers. Use add_layer() method to add layers sequentially.")
        else:
            assert (len(layer_sizes)) == (len(activations) + 1), "Provided number of layers does not have sufficient number of activation functions specified."

            self.layers = [Layer(index = i+1,
                                 input_size = layer_sizes[i],
                                 output_size = layer_sizes[i+1],
                                 weight_init=self.weight_init,
                                 activation = activations[i])
                                 for i in range(len(activations))]

    def add_layer(self, input_size: int, output_size: int, weight_init = "random", activation = "identity"):
        if self.layers is None:
            self.layers[Layer(index=1,
                              input_size=input_size,
                              output_size=output_size,
                              weight_init=weight_init,
                              activation=activation)]
        else:
            last_layer_index = self.layers[-1].index
            self.layers.append(Layer(index=last_layer_index+1,
                                     input_size=input_size,
                                     output_size=output_size,
                                     weight_init=weight_init,
                                     activation=activation))
    
    def forward(self, x: npt.NDArray) -> npt.NDArray:
        """
        Perform forward pass of `x` through the neural network.

        Takes the input `x` and passes through all of the neural network layers to come up with final output.

        Parameters
        ----------
        `x`: `NDArray`
            The input vector to feed to the network.
        
        Returns
        -------
        `NDArray`
            The activated output of the network after forward pass of `x`.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, y_real: npt.NDArray, y_pred: npt.NDArray):
        L = len(self.layers)
        db_list = deque()
        dW_list = deque()
        # Last layer's del_L/del_a for multiclass classification
        # Assuming that y_real will be one-hot vector for that class label
        da = y_pred - y_real
        
        for k in range(L-1, -1, -1):
            db = da # del_L/del_b for this layer
            db_list.appendleft(db)

            h_prev_layer = self.layers[k-1].h if k>0 else self.layers[k].x # previous layer's output is the initial input x if it is the first layer
            
            dW = da @ h_prev_layer.T # del_L_del_W for this layer
            dW_list.appendleft(dW)

            if k == 0:
                # since done computing dW and db for first (hidden) layer
                # no need to compute dh and and using that to
                # compute da for previous layer which is the input itself
                break

            # compute del_L/del_h for next iteration (/of previous layer)
            dh = self.layers[k].W.T @ da
            
            # compute del_L/del_a of previous layer
            da = dh * self.layers[k-1].activation.grad(self.layers[k-1].a)
        
        return (dW_list, db_list)
    
    def initialize(self, X_train: npt.NDArray, Y_train: npt.NDArray, batch_size = 1):
        assert (X_train.shape[0] == Y_train.shape[0]), "Number of training examples does not match with number of provided labels."
        self.X_train = X_train
        self.Y_train = Y_train
        self.batch_size = batch_size

        self.W_optimizers = deque()
        self.b_optimizers = deque()

        for layer in self.layers:
            self.W_optimizers.append(Optimizer(theta=layer.W, learning_rate=self.lr,
                                               weight_decay=self.wd, type=self.opt_name,
                                               **self.optimizer_params))
            self.b_optimizers.append(Optimizer(theta=layer.b, learning_rate=self.lr,
                                               weight_decay=self.wd, type=self.opt_name,
                                               **self.optimizer_params))
        
        print("Setup training data and initialized optimizers for the model.")

    def evaluate(self, X_valid: npt.NDArray, Y_valid: npt.NDArray):
        K = Y_valid.shape[1] # Number of classes
        
        total_valid_loss = 0
        cf_valid = np.zeros(shape=(K,K), dtype=np.int_) # Confusion matrix

        for x_valid, y_real_valid in zip(X_valid, Y_valid):
            # Do forward pass
            y_pred_valid = self.forward(x_valid)
            # Accumulate this validation examples loss
            total_valid_loss += losses.ce(y_real_valid, y_pred_valid)
            # Update the confusion matrix
            valid_pred_label, valid_real_label = confusion_labels(y_pred_valid, y_real_valid)
            cf_valid[valid_pred_label][valid_real_label] += 1
        
        valid_acc = accuracy(cf_valid) # Evaluate validation accuracy based on confusion matrix
        total_valid_loss /= Y_valid.shape[0] # Need to judge model on a loss per example basis

        print(f"valid_loss = {total_valid_loss:.5f}\tvalid_acc = {valid_acc:.5f}")
        return total_valid_loss, valid_acc, cf_valid

    def training_step(self):
        num_examples_seen = 0
        total_train_loss = 0
        L = 0
        total_db_list = deque()
        total_dW_list = deque()
        
        K = self.Y_train.shape[1] # Number of classes
        # Confusion matrix to be built over this epoch
        cf_train = np.zeros(shape=(K,K), dtype=np.int_)

        # Initialize the total parameter change variables
        for layer in self.layers:
            L += 1
            total_db_list.append(np.zeros_like(layer.b))
            total_dW_list.append(np.zeros_like(layer.W))
        
        # Pass over the training data once
        for x, y_real in zip(self.X_train, self.Y_train):
            # Forward pass to find y_pred
            y_pred = self.forward(x)
            assert (y_real.shape == y_pred.shape), "The training could not start as the prediction shape does not match true output."

            # Accumulate this example's loss
            total_train_loss += losses.ce(y_real, y_pred)

            pred_label, real_label = confusion_labels(y_pred=y_pred, y_real=y_real)
            cf_train[pred_label][real_label] += 1

            # Backpropagate to find dW and db for all layers for current example
            curr_dW_list, curr_db_list = self.backward(y_real, y_pred)

            # Accumulate this into total dW and db per batch
            for l in range(L):
                total_db_list[l] += curr_db_list[l]
                total_dW_list[l] += curr_dW_list[l]
            
            num_examples_seen += 1

            # If a batch of examples worth dW and db has been accumulated, take an optimization step
            if num_examples_seen%self.batch_size == 0:
                # Take a optimization step
                for l in range(L):
                    self.b_optimizers[l].update(total_db_list[l])
                    self.W_optimizers[l].update(total_dW_list[l])
                    # Reset the total dW and db for next batch
                    total_db_list[l] *= 0.0
                    total_dW_list[l] *= 0.0

        train_acc = accuracy(cf_train) # Evaluate training accuracy from confusion matrix
        total_train_loss /= self.Y_train.shape[0] # Need to judge model on a loss per example basis

        print(f"train_loss = {total_train_loss:.5f}\ttrain_acc = {train_acc:.5f}")
        return total_train_loss, train_acc, cf_train
