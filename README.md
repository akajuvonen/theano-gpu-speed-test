# theano-gpu-speed-test
Testing how much gpu speeds up calculations in a simple example.

## Instructions

### Install dependencies

NOTE: Tested on Ubuntu 17.04. Should work on modern Linux distributions.

First, you need Python and Conda. I recommend [MiniConda](https://conda.io/miniconda.html). In addition, [CUDA](https://developer.nvidia.com/cuda-downloads) and [CuDNN](https://developer.nvidia.com/cudnn) must be installed. You need a free NVidia developer account for CuDNN. Then just run `bin/init` to make a Conda virtual environment with required dependencies.

Configuring CUDA can sometimes be a bit of work. However, it must be done and everything has to work prior to running the scripts. On some Linux distributions you can just install a package from a repository, however.

[Theano](http://deeplearning.net/software/theano/index.html) by Conda with all the other dependencies. If you get any error messages, or using the GPU with Theano doesn't work, check and configure it separately.

### Run the analysis

Use `bin/run` to actually run the analysis. The init script must be run before this, otherwise it will just print an error.

The analysis performs an element-wise addition of two random matrices several times. After completion it will print the elapsed times in the following way:
```
----- RESULTS -----
Time used with CPU: 1.407127
Time used with GPU: 0.022818
GPU was 61.67 times faster
```

### Cleaning

In order to clean the virtual environment, use `bin/clean`. If you want to run the analysis again, use `bin/init` before that.
