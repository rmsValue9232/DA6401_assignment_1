import numpy as np
import numpy.typing as npt

class Activation():
    """
    Activation function constructor class.
    - Construct an Activation object with given `choice` of the activation function.
    - Use activation function by calling method `myActivation.function(x)` on `x`.
    - `x` is expected to be of type numpy `float` or `NDArray`.

    Attributes
    ----------
    `choice` : `str`, ["identity", "sigmoid", "tanh", "relu", "softmax"], default = "identity"
        The choice of the activation function to use.
    """
    _valid_choices = ["identity", "sigmoid", "tanh", "relu", "softmax"]
    
    def __init__(self, choice = "identity"):
        if choice not in self._valid_choices:
            raise ValueError(f"Invalid activation choice '{choice}', choose from: {self._valid_choices}, {len(self._valid_choices)}")
        
        self.choice = choice
        self.function = self.identity
        self.grad = self.identity_grad
        
        if self.choice == "softmax":
            self.function = self.softmax
        elif self.choice == "sigmoid":
            self.function = self.sigmoid
            self.grad = self.sigmoid_grad
        elif self.choice == "tanh":
            self.function = self.tanh
            self.grad = self.tanh_grad
        elif self.choice == "relu":
            self.function = self.relu
            self.grad = self.relu_grad

    
    def identity(self, x: float | npt.NDArray) -> float | npt.NDArray:
        """
        Performs identity activation on provided input.

        Parameters
        ----------
        `x` : `float | NDArray`
            Provided input.
        
        Returns
        -------
        `float | NDArray`
            Activation output.
        """
        return x
    
    def sigmoid(self, x: float | npt.NDArray) -> float | npt.NDArray:
        """
        Performs sigmoid activation on provided input.

        Parameters
        ----------
        `x` : `float | NDArray`
            Provided input.
        
        Returns
        -------
        `float | NDArray`
            Activation output.
        """
        return np.where(x>=0,
                        1.0/(1.0 + np.exp(-x)),
                        np.exp(x)/(1.0 + np.exp(x)))

    def tanh(self, x: float | npt.NDArray) -> float | npt.NDArray:
        """
        Performs tanh activation on provided input.

        Parameters
        ----------
        `x` : `float | NDArray`
            Provided input.
        
        Returns
        -------
        `float | NDArray`
            Activation output.
        """
        return np.tanh(x)

    def relu(self, x: float | npt.NDArray) -> float | npt.NDArray:
        """
        Performs ReLU activation on provided input.

        Parameters
        ----------
        `x` : `float | NDArray`
            Provided input.
        
        Returns
        -------
        `float | NDArray`
            Activation output.
        """
        return np.maximum(0, x)
    
    def softmax(self, x: float | npt.NDArray) -> float | npt.NDArray:
        """
        Performs softmax activation on provided input.

        Utilises log-sum-exp trick to prevent overflow due to large exponents.

        Parameters
        ----------
        `x` : `float | NDArray`
            Provided input.
        
        Returns
        -------
        `float | NDArray`
            Activation output.
        """
        if isinstance(x, np.ndarray):
            x_max = np.max(x)
            exp_x = np.exp(x - x_max)
            result = exp_x/np.sum(exp_x)
        else:
            result = 1.0

        return result
    
    def identity_grad(self, x: float | npt.NDArray) -> float | npt.NDArray:
        return 1.0
    
    def sigmoid_grad(self, x: float | npt.NDArray) -> float | npt.NDArray:
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def tanh_grad(self, x: float | npt.NDArray) -> float | npt.NDArray:
        return (1.0 - self.tanh(x)**2)
    
    def relu_grad(self, x: float | npt.NDArray) -> float | npt.NDArray:
        if isinstance(x, np.ndarray):
            return np.float_(x > 0)
        else:
            return float(x > 0)
    
    