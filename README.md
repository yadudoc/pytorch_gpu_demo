# Using GPUs effectively: A tutorial with Pytorch

This tutorial with use Parsl: Parallel programming library for Python to:
1. Request GPU nodes from the fe.cs.uchicago cluster
2. Demo launching a Pytorch based MNIST training example in various GPU configurations
3. Show checkpoint and restart functionality.


## Setup your environment

fe.cs.uchicago cluster comes with a conda installation already available. We'll be using this
to install the requirements for the demo.

```
$ conda create --yes --name parsl_py3.7 python=3.7
$ conda activate parsl_py3.7
$ pip install parsl==1.0.0
$ conda install -c torch torchvision
```





