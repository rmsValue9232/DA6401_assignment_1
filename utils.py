import numpy as np
import numpy.typing as npt

def one_hot_encoder(y_label_batch: npt.NDArray):
    """
    y_label_batch is batch of labels in a numpy array
    hence expect y_label_batch.ndim = 1
    """
    N = y_label_batch.shape[0] # total number of examples
    D = np.unique(y_label_batch).shape[0] # total number of unique labels

    Y = np.zeros(shape=(N, D, 1))
    for i in range(N):
        Y[i][y_label_batch[i]][0] = 1.0
    
    return Y

def flatten_image_to_vector(x_image_batch: npt.NDArray):
    """
    x_image_batch is a batch of images in numpy array
    hence expect x_image_batch.ndim = 3

    first dim size gives batch size
    """
    batch_shape = x_image_batch.shape
    N = batch_shape[0]
    H = batch_shape[1]
    W = batch_shape[2]
    F = H*W
    X = np.zeros(shape=(N, F, 1))
    for i in range(N):
        X[i] = x_image_batch[i].reshape((F, 1))
    
    return X

def train_valid_split(x, y, train_percent=0.9):
    """
    Assuming first dimension/axis of the data input output pairs to be the batch size 
    split the data into training and validation batches.
    This does not do randomization yet.
    """
    assert (x.shape[0] == y.shape[0]), "batch size of the input and outputs do not match, so cannot proceed."
    assert (train_percent > 0 and train_percent < 1), "keep the train size percentage in range (0, 1)"
    N = x.shape[0]
    train_size = round(train_percent * N)
    x_train, x_valid = x[:train_size], x[train_size:]
    y_train, y_valid = y[:train_size], y[train_size:]

    return (x_train, y_train), (x_valid, y_valid)