pix2pix-pytorch
PyTorch implementation of Image-to-Image Translation Using Conditional Adversarial Networks.

Based on pix2pix by Phillip Isola et al.

The examples from the paper:

examples

Prerequisites
Linux
Python, Numpy, PIL
pytorch 1.5.0
cuda 10.1
torchvision 0.6.0

Train the model:
python train_CUFS.py

Test the model:
python test.py

Acknowledgments
This code is a simple implementation of pix2pix. Easier to understand. Note that we use a downsampling-resblocks-upsampling structure instead of the unet structure in this code, therefore the results of this code may inconsistent with the results presented in the paper.

Highly recommend the more sophisticated and organized code pytorch-CycleGAN-and-pix2pix by Jun-Yan Zhu.
