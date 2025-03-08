import numpy as np
import numpy.typing as npt

def mse(y_real: npt.NDArray, y_pred: npt.NDArray) -> float:
    """
    Evaluates mean squared error for provided actual response vs predicted response.

    Args:
        y_real (NDArray): Actual response
        y_pred (NDArray): Predicted response
    
    Returns:
        mse (float): The mean squared error
    """
    mse = np.mean((y_pred - y_real)**2)
    return float(mse)

def ce(y_real: npt.NDArray, y_pred: npt.NDArray) -> float:
    """
    Evaluates cross entropy for provided actual response distribution vs predicted response distribution.

    Args:
        y_real (NDArray): Actual response distribution
        y_pred (NDArray): Predicted response distribution
    
    Returns:
        ce (float): The cross entropy between the two distributions
    """
    assert (np.min(y_real) >= 0.0 and np.max(y_real)<=1.0 and np.sum(y_real) == 1.0), "y_real is not a valid probability distribution"
    assert (np.min(y_pred) >= 0.0 and np.max(y_pred)<=1.0 and np.sum(y_pred) == 1.0), "y_pred is not a valid probability distribution"

    # avoids getting log(0.0) in calculations
    y_pred = np.clip(y_pred, 1e-12, 1.0 - 1e-12)

    ce = - np.sum(y_real * np.log(y_pred))
    return float(ce)
