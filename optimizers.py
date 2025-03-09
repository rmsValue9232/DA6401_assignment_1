import numpy as np
import numpy.typing as npt

class Optimizer():
    _optimizer_choices = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
    def __init__(self, theta: float | npt.NDArray, learning_rate: float, type="sgd", **kwargs):
        if type not in self._optimizer_choices:
            raise ValueError(f"Unknown optimizer type: {type}, choose from: {self._optimizer_choices}")
        
        # Initiate theta_0
        self.theta = theta
        self.type = type
        self.learning_rate = learning_rate
        self.extra_params = kwargs
        print(self.extra_params)

        # Initiate v_0 = 0
        self.v = np.zeros_like(self.theta) if isinstance(self.theta, np.ndarray) else 0.0
        # Initiate time t_0 = 0
        self.t = 0
        # Initiate moment m_0 = 0
        self.m = np.zeros_like(self.theta) if isinstance(self.theta, np.ndarray) else 0.0
        
    def update(self, dTheta: float | npt.NDArray):
        self.t += 1
        if self.type == "sgd":
            self.theta -= self.learning_rate * dTheta
            
        elif self.type == "momentum":
            assert "beta" in self.extra_params.keys(), (f"Argument beta not supplied for optimizer with provided optimizer type as momentum.")
            
            beta = self.extra_params["beta"]
            assert beta >= 0 and beta < 1, (f"beta should be in interval[0, 1), but provided beta is {beta}.")
            
            self.v = (beta * self.v) - (self.learning_rate * dTheta)

            self.theta += self.v

        elif self.type == "nag":
            assert "beta" in self.extra_params.keys(), (f"Argument beta not supplied for optimizer with provided optimizer type as NAG.")

            beta = self.extra_params["beta"]
            assert beta >= 0 and beta < 1, (f"beta should be in interval[0, 1), but provided beta is {beta}.")

            self.v = (beta * self.v) - (self.learning_rate * dTheta)

            self.theta += (beta * self.v) - (self.learning_rate * self.theta)
        
        elif self.type == "rmsprop":
            assert "beta" in self.extra_params.keys(), (f"Argument beta not supplied for optimizer with provided optimizer type as RMSprop.")
            assert "epsilon" in self.extra_params.keys(), (f"Argument epsilon not supplied for optimizer with provided optimizer type as RMSprop.")

            epsilon = self.extra_params["epsilon"]
            beta = self.extra_params["beta"]
            assert beta >= 0 and beta < 1, (f"beta should be in interval[0, 1), but provided beta is {beta}.")

            self.v = (beta * self.v) + ((1-beta) * dTheta**2)
            effective_lr = self.learning_rate/np.sqrt(self.v + epsilon)
            self.theta -= effective_lr * dTheta

        elif self.type == "adam":
            assert "beta1" in self.extra_params.keys(), (f"Argument beta1 not supplied for optimizer with provided optimizer type as adam.")
            assert "beta2" in self.extra_params.keys(), (f"Argument beta2 not supplied for optimizer with provided optimizer type as adam.")
            assert "epsilon" in self.extra_params.keys(), (f"Argument epsilon not supplied for optimizer with provided optimizer type as adam.")

            epsilon = self.extra_params["epsilon"]
            beta1 = self.extra_params["beta1"]
            assert beta1 >= 0 and beta1 < 1, (f"beta1 should be in interval[0, 1), but provided beta1 is {beta1}.")
            beta2 = self.extra_params["beta2"]
            assert beta2 >= 0 and beta2 < 1, (f"beta2 should be in interval[0, 1), but provided beta2 is {beta2}.")

            self.m = (beta1 * self.m) + ((1-beta1) * dTheta)
            self.v = (beta2 * self.v) + ((1-beta2) * dTheta**2)

            m_hat = self.m/(1 - beta1**self.t)
            v_hat = self.v/(1 - beta2**self.t)
            effective_lr = self.learning_rate/(epsilon + np.sqrt(v_hat))
            self.theta -= effective_lr * m_hat

        elif self.type == "nadam":
            assert "beta1" in self.extra_params.keys(), (f"Argument beta1 not supplied for optimizer with provided optimizer type as Nadam.")
            assert "beta2" in self.extra_params.keys(), (f"Argument beta2 not supplied for optimizer with provided optimizer type as Nadam.")
            assert "epsilon" in self.extra_params.keys(), (f"Argument epsilon not supplied for optimizer with provided optimizer type as Nadam.")

            epsilon = self.extra_params["epsilon"]
            beta1 = self.extra_params["beta1"]
            assert beta1 >= 0 and beta1 < 1, (f"beta1 should be in interval[0, 1), but provided beta1 is {beta1}.")
            beta2 = self.extra_params["beta2"]
            assert beta2 >= 0 and beta2 < 1, (f"beta2 should be in interval[0, 1), but provided beta2 is {beta2}.")

            self.m = (beta1 * self.m) + ((1-beta1) * dTheta)
            self.v = (beta2 * self.v) + ((1-beta2) * dTheta**2)

            m_hat = self.m/(1 - beta1**(self.t))
            v_hat = self.v/(1 - beta2**(self.t))

            effective_lr = self.learning_rate/(epsilon + np.sqrt(v_hat))

            self.theta -= effective_lr * ( (beta1*self.m) + (dTheta * (1 - beta1)/(1 - beta1**self.t)) )

