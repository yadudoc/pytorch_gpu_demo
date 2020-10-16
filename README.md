# Using GPUs effectively: A tutorial with Pytorch

This tutorial with use Parsl: Parallel programming library for Python to:

1. Request GPU nodes from the **fe.ai.cs.uchicago.edu** cluster
2. Demo launching a Pytorch based MNIST training example in various GPU configurations
3. Show checkpoint and restart functionality.


## Setup your environment

**fe.ai.cs.uchicago.edu** cluster comes with a conda installation already available. We'll be using this
to install the requirements for the demo.

```
$ conda create --yes --name parsl_py3.7 python=3.7
$ conda activate parsl_py3.7
$ pip install parsl==1.0.0
$ conda install -c torch torchvision
```

Let's check the environment sanity with a basic mnist application:

```
# This should print a sequence of results from training, this will be slow running on the login node
(parsl_py3.7) $ python3 torch_mnist.py --epochs=1

# Let's config parsl is installed. This should print 1.0.0
(parsl_py3.7) $ python3 -c "import parsl; print(f'Parsl version: {parsl.__version__}')"
```

## Parsl Configuration

A sample configuration file `config.py` contains a config object that requests nodes
from the the slurm scheduler and launches 1 manager+worker pair for each available GPU.

**Note**: The `SrunLauncher.overrides` feature is used here to *hack* the launcher into launching more manager+worker groups per node than usual. Usually 1 manager manages the whole node, but in this situation we want the manager and it's child processes (workers) to be bound to each GPU.

Please tune the config via the variables :

```
# Configure options here:
NODES_PER_JOB = 2
GPUS_PER_NODE = 4 
GPUS_PER_WORKER = 1
```

## Basic Grid Search

The `basic_grid_search.py` example shows you the following :

1. Running a very simple `python_app` called `platinfo` that returns the nodename and CUDA information
2. `run_mnist` takes a range of batch sizes and epochs, and launches the `torch_mnist` application
3. The `torch_mnist` application is a `bash_app` that invokes the `torch_mnist.py` example from [pytorch examples](https://github.com/pytorch/examples/tree/master/mnist) from the commandline on each worker which is bound to 1 GPU on the cluster nodes.

## Checkpointed runs

The `checkpoint_test.py` along with `torch_mnist_checkpointed.py` shows you how to run torch applications with checkpoint and restart functionality.

The key updates to `torch_mnist_checkpointed.py` are:
1. Updated `train_model` method that takes proper python params rather than argparge.args
2. `checkpoint_period` kwarg option that specifies how many minutes apart checkpoint events should be triggered.
3. `checkpoint_input` and `checkpoint_output` paths that define paths from/to which checkpoints should be read/written.
4. Minor code blocks that load and write checkpoints.

`checkpoint_test.py` uses a `python_app` that explicitly adds the current directory to the module path,
so that the methods in the `torch_mnist_checkpointed` module can be imported on the compute node which don't share the python environment on the login node. This test sets a low walltime so that workers and their mnist training tasks are terminated due to node loss to simulate a failure. The test sets `config.retries=3` so that the application is rerun and with checkpoint restart support, very little compute is lost.





