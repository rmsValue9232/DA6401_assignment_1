import wandb
from nn import NeuralNetwork
import utils
import argparse

sweep_configuration = {
    "name": "sweeptry",
    "method": "bayes",
    "metric": {
        "goal": "maximize",
        "name": "valid_acc"
    },
    "parameters": {
        "learning_rate": {"min": 1.0e-4,"max": 0.1},
        "batch_size": {"values": [8, 16, 32, 64]},
        "epochs": {"values": [3]},
        "optimizer": {"values": ["sgd", "momentum", "nag"]},
        "beta": {"min":0.0, "max": 0.9},
        "beta1":{"min":0.0, "max": 0.9},
        "beta2":{"min":0.0, "max": 0.9},
        "epsilon": {"values": [1.e-12, 1.e-8, 1.e-4]},
        "weight_init": {"values": ["random", "xavier"]},
        "weight_decay": {"values": [1e-4, 1e-5, 1e-6, 1e-7]},
        "num_layers": {"values": [1,2,3]},
        "hidden_size": {"values": [64, 32, 16]},
        "activation": {"values": ["identity", "sigmoid", "tanh", "relu"]}
    }
}

def sweep():
    run = wandb.init(project="my-first-sweep")
    config = wandb.config

    lr = config.learning_rate
    bs = config.batch_size
    epochs = config.epochs
    opt_name = config.optimizer
    beta = config.beta
    beta1= config.beta1
    beta2= config.beta2
    epsilon= 1e-6
    weight_init = config.weight_init
    weight_decay = config.weight_decay
    num_layers = config.num_layers
    hidden_size = config.hidden_size
    activation = config.activation

    run_name = f"hl_{num_layers}_bs_{hidden_size}_ac_{activation}"
    run.name = run_name
    

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = utils.load_data("fashion_mnist")

    X_train = utils.standardScale(utils.flatten_image_to_vector(x_train))
    X_valid = utils.standardScale(utils.flatten_image_to_vector(x_valid))
    X_test  = utils.standardScale(utils.flatten_image_to_vector(x_test))

    Y_train = utils.one_hot_encoder(y_train)
    Y_valid = utils.one_hot_encoder(y_valid)
    Y_test  = utils.one_hot_encoder(y_test)
    
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

    # Add the hidden layers based on config
    layer_sizes = [num_input_f]
    layer_sizes += [hidden_size]*num_layers

    # Collect the optimizer's params
    optimizer_params = dict()
    if opt_name == "momentum":
        optimizer_params["beta"] = beta
    elif opt_name == "nag":
        optimizer_params["beta"] = beta
    elif opt_name == "rmsprop":
        optimizer_params["beta"] = beta
        optimizer_params["epsilon"] = epsilon
    elif opt_name == "adam":
        optimizer_params["beta1"] = beta1
        optimizer_params["beta2"] = beta2
        optimizer_params["epsilon"] = epsilon
    elif opt_name == "nadam":
        optimizer_params["beta1"] = beta1
        optimizer_params["beta2"] = beta2
        optimizer_params["epsilon"] = epsilon
    
    # Form the neural network
    model = NeuralNetwork(weight_init=weight_init,
                        layer_sizes=layer_sizes,
                        activations=[activation]*num_layers,
                        optimizer=opt_name,
                        learning_rate=lr,
                        weight_decay=weight_decay,
                        **optimizer_params)
    
    # Add the output "prediction" layer with softmax activation
    model.add_layer(input_size=layer_sizes[-1],
                    output_size=len(class_images),
                    weight_init=weight_init,
                    activation="softmax")
    
    # Initialize the model with training dataset and setup its optimizer
    model.initialize(X_train, Y_train)
    for epoch in range(epochs):
        total_train_loss, train_acc, cf_train = model.training_step()
        total_valid_loss, valid_acc, cf_valid = model.evaluate(X_valid, Y_valid)
        print(f"Completed epoch {epoch}")
        log_sheet = {
            "train_loss": round(total_train_loss, 4),
            "train_acc" : round(train_acc, 4),
            "valid_loss": round(total_valid_loss, 4),
            "valid_acc" : round(valid_acc, 4)
        }

        wandb.log(log_sheet)

def hyperparameter_sweep():
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="new-sweep")
    wandb.agent(sweep_id, function=sweep, count=3)
    wandb.finish()

def get_args():
    parser = argparse.ArgumentParser()
    
    # WandB arguments
    parser.add_argument("-wp", "--wandb_project", type=str, default="", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", type=str, default="", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    
    # Training arguments
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset to use for training")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs to train neural network")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size used to train neural network")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd", help="Optimizer to use")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1, help="Learning rate used to optimize model parameters")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum used by momentum and nag optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta used by rmsprop optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5, help="Beta1 used by adam and nadam optimizers")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5, help="Beta2 used by adam and nadam optimizers")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon used by optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay used by optimizers")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="random", help="Weight initialization method")
    
    # Model arguments
    parser.add_argument("-nhl", "--num_layers", type=int, default=1, help="Number of hidden layers in feedforward neural network")
    parser.add_argument("-sz", "--hidden_size", type=int, default=4, help="Number of hidden neurons in a feedforward layer")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid", help="Activation function")
    
    return parser.parse_args()

def main(args):
    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity
    ) as run:
        pass

if __name__ == "__main__":
    args = get_args()
    main(args)
    # For hyperparameter sweep
    # hyperparameter_sweep()

    




# config = {
#     "dataset": "fashion_mnist",
#     "epochs": 3,
#     "batch_size": 16,
#     "loss": "cross_entropy",
#     "optimizer": "adam",
#     "learning_rate": 0.001,
#     "beta": 0.5,
#     "beta1": 0.5,
#     "beta2": 0.5,
#     "epsilon": 1e-6,
#     "weight_decay": 0.0005,
#     "weight_init": "xavier",
#     "num_layers": 2,
#     "hidden_size": 16,
#     "activation": "tanh"
# }
