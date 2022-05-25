# On the Maximum Hessian Eigenvalue and Generalization

This repository is the official implementation of the NeurIPS 2022 submission **On the Maximum Hessian Eigenvalue and Generalization**.

## Preliminaries
Run the following command to install the requirements:
```setup
pip install -r src/requirements.txt
```
Next, set the `DATASETS` and `RESULTS` environments to the directories where the datasets and results will be stored, respectively (e.g., `export DATASET="/home/userID/datasets"`).

Finally, preprocess and save the datasets by run the following command:
```setup
python src/preprocess_data.py
```

## Training

The script `train.py` trains a neural network using minibatch stochastic gradient descent (SGD). 
The required arguments are:
```setup
train.py [dataset] [arch_id] [batch_size] [lr]
```


For example, running the command below trains a VGG11 (no batch-normalization, no dropout) on CIFAR-10 using cross-entropy loss, batch size 100, and learning rate 0.01. Training terminates once the train accuracy reaches 99%:
```setup
python src/train.py 'cifar10' 'vgg11_no_dropout' 100 1e-1 --loss='ce'
```

The corresponding training results are saved in the following output directory:
```setup
${RESULTS}/cifar10/arch_vgg11_no_dropout/ce/sgd/no_change_lr/lr_0.1_bs_100_0
```

The following files are created inside this output directory:
- `train_loss`, `test_loss`, `train_acc`, `test_acc`: the train and test losses and accuracies (recorded after each epoch)
- `leading_eigenvalues/leading_eigenvalue_final`: the maximum Hessian eigenvalue (calculated at the final iterate; for batch-normalized networks trained at small learning rates, this is calculated between iterates)

## Evaluation

The script `eval.py` loads a trained network from its final checkpoint, and prints (i) the final test accuracy and (ii) the maximum Hessian eigenvalue.

Running `eval.py` is analagous to running `train.py`, where the required arguments are:
```setup
eval.py [dataset] [arch_id] [batch_size] [lr]
```

## Complete documentation
The script `train.py` trains a neural network using minibatch stochastic gradient descent (SGD). 
The script `eval.py` loads a trained network from its final checkpoint, and prints (i) the final test accuracy and (ii) the maximum Hessian eigenvalue.

The required arguments of `src/train.py` and `src/eval.py` are:
- `dataset` [string]: the dataset to train on. Possible values are listed below:
  - `cifar10`: the full CIFAR-10 dataset
  - `fashion_mnist`: the full FashionMNIST dataset
  - `cifar10_1k`: the first 1000 examples from the full CIFAR-10 dataset
  - `fashion_mnist_1k`:  the first 1000 examples from the full FashionMNIST dataset
- `arch_id` [string]: the neural network achitecture  (see `load_architecture()` in `src/architectures.py` for possible values)
- `batch_size` [int]: the batch size
- `lr` [float]: the learning rate

The optional arguments of `src/train.py` and `src/eval.py` are
- `opt` [str, defaults to `sgd`]: the optimization algorithm to use during training
  - `sgd`: vanilla SGD
  - `sam`: Sharpness Aware Minimization (see for the [original paper](https://arxiv.org/abs/2010.01412) and the [PyTorch implementation](https://github.com/davda54/sam) for more details) 
- `rho` [float, defaults to `0.05`]: SAM hyperparameter (note this is only applicable when the optimizer is `sam`)
- `loss` [str, defaults to `ce`]: the loss function (as of now, only cross-entropy loss `ce` is supported)
- `network_seed` [int, defaults to `0`]: the random seed used when initializing the network weights
- `train_seed` [int, defaults to `1`]: the random seed used at the start of training
- `acc_goal` [float, defaults to `0.99`]: terminate training if the train accuracy ever crosses this value
- `physical_batch_size` [int, defaults to `1000`]: the maximum number of examples that we try to fit on the GPU at once
- `abridged_size` [int, defaults to `1000`]: when computing the maximum Hessian eigenvalue, use an abridged dataset of this size
- `disable_dropout` [bool, defaults to `False`]: `True` when computing the maximum Hessian eigenvalue using a network with dropout layers (`False` otherwise)
- `between_iterates` [bool, defaults to `False`]: `True` when computing the maximum Hessian eigenvalue between iterates (only used for batch-normalized networks trained with small learning rates)
- `save_freq`: [int, defaults to `10`]: the frequency at which we save results
