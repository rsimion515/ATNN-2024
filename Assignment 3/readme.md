Assignment 3

Simion Ruben-Andrei MISS12

Colab documentation: https://colab.research.google.com/drive/1mBDZQ-vdSc8WbtVGCUBZE58shWTkHuY0?usp=sharing


The Generic Pipeline support the following:

1. It is device agnostic, works on both CPU and GPU.

2. Can be configured to use the following datasets: MNIST, CIFAR-10 and CIFAR-100.

3. Datasets are cached locally.

4. Configurable DataLoaders for training and testing.

5. Can be configured to use any model (if implemented). Supports the loading:

• For CIFAR: resnet18 cifar10 from timm and PreActResNet-18 (from Lab 2) and VGG16 (from Assignment 2).

• For MNIST: MLP and LeNet.

6. Can be configured to use optimizers such as SGD, SGD with momentum, SGD with nesterov, SGD with weight decay, Adam, AdamW, RmsProp.

7. Can be configured to use any implemented LR scheduler: StepLR and ReduceLROnPlateau.

8. Supports an early stopping mechanism.

9. Provides the option to select different data augmentation schemes.
