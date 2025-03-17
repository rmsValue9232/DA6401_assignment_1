import numpy as np
import numpy.typing as npt
from keras.datasets import fashion_mnist, mnist

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

def confusion_labels(y_pred: npt.NDArray, y_real: npt.NDArray):
    assert (y_pred.shape == y_real.shape), "Shape of y_pred and y_real does not match."

    assert (y_pred.ndim == 2), "Not a valid prediction column vector of probabilities."

    y_pred = np.int8((y_pred == np.max(y_pred)))
    pred_label = np.argwhere(y_pred == 1)[0][0]
    real_label = np.argwhere(y_real == 1)[0][0]

    return (pred_label, real_label)

def accuracy(confusion_matrix: npt.NDArray):
    dims = confusion_matrix.ndim
    shap = confusion_matrix.shape
    assert (dims == 2), "Input is not a 2-dim matrix."
    assert (shap[0] == shap[1]), "Input is not a square matrix"
    total_examples = np.sum(confusion_matrix)
    correct_examples = np.sum(np.diag(confusion_matrix))

    return float(correct_examples)/float(total_examples)

def load_data(name: str):
    # Choose the data to load based on config
    if name == "fashion_mnist":
        data_module = fashion_mnist
    elif name == "mnist":
        data_module = mnist
    
    # Load the data: train test split done automatically
    (x_train, y_train), (x_test, y_test) = data_module.load_data()
    print("Loaded the train and test data.")
    # reserve small portion of train set for validation
    (x_train, y_train), (x_valid, y_valid) = train_valid_split(x_train, y_train, train_percent=0.9)
    print("Perfomed train validation split.")

    print(f"x_train.shape = {x_train.shape},\ty_train.shape = {y_train.shape}")
    print(f"x_valid.shape = {x_valid.shape},\t\ty_valid.shape = {y_valid.shape}")
    print(f"x_test.shape = {x_test.shape},\t\ty_test.shape = {y_test.shape}")

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)