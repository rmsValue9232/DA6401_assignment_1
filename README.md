# Assignment 1 - Introduction to Deep Learning (DA6401)

- Submittted by: Aditya Kumar
- Roll No.: MA24M001
- Course Instructor: Dr. Mitesh Khapra
- Submission Date: 17 March, 2025

---

- Github Repo - <https://github.com/rmsValue9232/DA6401_assignment_1>
- `losses.py` contain MSE and CE loss functions.
- `optimizers.py` contain `Optimizer` class that builds the optimizer object for model parameter tuning.
- `utils.py` contains:
  - One Hot Encoder for labels
  - Image to flat feature vector
  - Train valid test set splitter
  - Confusion matrix indices (of the example label and its predicted value) finder
  - Accuracy finder
  - Data Loader based on the data name passed.
  - Standard scaler for examples.
- `activations.py` builds the various activation functions and their gradients (except softmax).
- `nn.py` builds two classes:
  - `Layer` which in the NN architecture sort of sits in between two layers in usual lingo. It contains the weights and biases that will be use to compute front layer's input from the previous layer's output.
  - `NeuralNetwork` builds the unifying achitecture:
    - It contains all the layers, their optimizers, the instantiation parameters of any model.
    - It contains forward pass that computes the final layer output and backward pass that computes the parameter gradients.
    - The gradients are then used by the layer optimizers to update the parameters.
    - It also includes one epoch training step routine, a evaluation function to evaluate model metrics on validation set.
- `train.py` provides functionalities to:
  - perform WandB hyperparameter sweep
  - conduct a single experiment run via argparsed configuration.

---

- `test.ipynb`, `test2.ipynb`, `test3.ipynb` are all notebooks where I have extensively tested the various modules I have built.

---
**Self Declaration**
I, Aditya_Kumar_MA24M001, swear on my honour that I have written the code and the report by myself and have not copied it from the internet or other students.
