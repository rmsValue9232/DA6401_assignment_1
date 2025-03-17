import wandb
from nn import NeuralNetwork
import json
import utils

config = {
    "dataset": "fashion_mnist",
    "epochs": 3,
    "batch_size": 16,
    "loss": "cross_entropy",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "beta": 0.5,
    "beta1": 0.5,
    "beta2": 0.5,
    "epsilon": 1e-6,
    "weight_decay": 0.0005,
    "weight_init": "xavier",
    "num_layers": 2,
    "hidden_size": 16,
    "activation": "tanh"
}

wandb.login()

run_name = f"hl_{config['num_layers']}_bs_{config['hidden_size']}_ac_{config['activation']}"

run = wandb.init(
    project="trying_my_my",
    config=config,
    name=run_name
)

(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = utils.load_data(config["dataset"])

# Flatten the examples and one-hot encode labels
# to be digestible to the Neural Network

X_train, Y_train = utils.flatten_image_to_vector(x_train), utils.one_hot_encoder(y_train)
print(f"X_train.shape = {X_train.shape}, Y_train.shape = {Y_train.shape}")

X_valid, Y_valid = utils.flatten_image_to_vector(x_valid), utils.one_hot_encoder(y_valid)
print(f"X_valid.shape = {X_valid.shape}, Y_valid.shape = {Y_valid.shape}")

X_test, Y_test = utils.flatten_image_to_vector(x_test), utils.one_hot_encoder(y_test)
print(f"X_test.shape = {X_test.shape}, Y_test.shape = {Y_test.shape}")

# Collect an image per class for all classes in a run

class_images = {}

for image, label in zip(x_train, y_train):
    if label not in class_images:
        class_images[label] = image
    if len(class_images) == 10:
        break

wandb.log({
    "Example_per_class": [
        wandb.Image(
            class_images[label],
            caption=f"Class: {label}"
        )
        for label in class_images.keys()
    ]
})

# Number of input layer features
num_input_f = X_train.shape[1]

# Build the model's layer struture
# Identify the input layer
layer_sizes = [num_input_f]
# Add the hidden layers based on config
num_hid = config["num_layers"]
layer_sizes += [config["hidden_size"]]*num_hid

# Collect the optimizer's params
optimizer_params = dict()
if config["optimizer"] == "momentum":
    optimizer_params["beta"] = config["beta"]
elif config["optimizer"] == "nag":
    optimizer_params["beta"] = config["beta"]
elif config["optimizer"] == "rmsprop":
    optimizer_params["beta"] = config["beta"]
    optimizer_params["epsilon"] = config["epsilon"]
elif config["optimizer"] == "adam":
    optimizer_params["beta1"] = config["beta1"]
    optimizer_params["beta2"] = config["beta2"]
    optimizer_params["epsilon"] = config["epsilon"]
elif config["optimizer"] == "nadam":
    optimizer_params["beta1"] = config["beta1"]
    optimizer_params["beta2"] = config["beta2"]
    optimizer_params["epsilon"] = config["epsilon"]

# Form the neural network
model = NeuralNetwork(weight_init=config["weight_init"],
                      layer_sizes=layer_sizes,
                      activations=[config["activation"]]*num_hid,
                      optimizer=config["optimizer"],
                      learning_rate=config["learning_rate"],
                      weight_decay=config["weight_decay"],
                      **optimizer_params)

# Add the output "prediction" layer with softmax activation
model.add_layer(input_size=layer_sizes[-1],
                output_size=len(class_images),
                weight_init=config["weight_init"],
                activation="softmax")

# Print the neural network architecture
for i in range(len(model.layers)):
    print(f"Layer_{i} has input neurons size = {model.layers[i].input_size}, has W.shape = {model.layers[i].W.shape}")
    print(f"Layer_{i} has output neurons size = {model.layers[i].output_size}")

model.train(X_train=X_train.copy(),
          Y_train=Y_train.copy(),
          X_valid=X_valid.copy(),
          Y_valid=Y_valid.copy(),
          batch_size=20, epochs=3)

wandb.finish()