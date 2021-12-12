# Semantic-Segmentation Approach using 3 different Architectures (UNET, Pix2Pix, DeepLabv3)

Initial Steps as required to get started with the Model Training:

# Image and Label Preprocessing

1) We first Import necessary library and modules to get started with Labels and input image Preprocessing such as cv2, matplotlib.pyplot as plt, PIL, numpy, pandas

2) Convert Json file as provided to us for training into png.

3) Break input image and labels of size 1024x1024 into 256x256
 
4) Now we convert images and labels into numpy array which is as required to feed in our Architecture. 

5) Here We had defined the Separate class for separate channels and total number of class for pixel vice classification is 4.

# Architecture: UNET

# About the Architecture:

This architecture contains two paths. First path is the contraction path (also called as the encoder) which is used to capture the context in the image.

The encoder is just a traditional stack of convolutional and max pooling layers. 

The second path is the symmetric expanding path (also called as the decoder) which is used to enable precise localization using transposed convolutions.

Thus it is an end-to-end fully convolutional network (FCN), i.e. it only contains Convolutional layers and does not contain any Dense layer because of which it can accept image of any size.

# Approach:

1) We had used keras Framework and initially get started by reading the numpy array for images and labels.

2) Used Train-Test-Validation and splitted over data in accordance with that.

3) For Model Evaluation we had used metrics such as f1 score and received the value of 0.6403, recall: 0.5968, precision: 0.6933 and accuracy value of 0.2767

4) Hyper Parameters tuning along with FocalTverskyLoss for dealing with imbalanced classes 

5) Defining model: output 4 channel

6) loss: binary cross entropy loss with value of 0.0973 and mean_iou: 0.4656

7) Final selected Parameters which shows best results are: epoch 30, batchsize:64   

# Architecture: Pix2Pix

# About the Architecture:

Generative Adversarial Network called Conditional Generative Adversarial Network to solve the problems in the successful training of the GAN.

Image to Image translation is one of the tasks, which can be done by Conditional Generative Adversarial Networks (CGANs) ideally.

In the task of Image to Image translation, an image can be converted into another one by defining a loss function which is extremely complicated. Accordingly, this task has many applications like colorization and making maps by converting aerial photos. 

Pix2Pix network was developed based on the CGAN. Some of the applications of this efficient method include object reconstruction from edges, photos synthesis from label maps, and image colorization 

Pix2Pix is based on conditional generative adversarial networks (CGAN) to learn a mapping function that maps an input image into an output image. Pix2Pix like GAN, CGAN is also made up of two networks, the generator and the discriminator.

The generator goal is to take an input image and convert it into the desired image (output or ground truth) by implementing necessary tasks. There are two types of the generator, including encoder-decoder and U-Net network.

The task of the discriminator is to measure the similarity of the input image with an unknown image. This unknown image either belongs to the dataset (as a target image) or is an output image provided by the generator.

The PatchGAN discriminator in Pix2Pix network is employed as a unique component to classify individual (N x N) patches within the image as real or fake.

Since the number of PatchGAN discriminator parameters is very low, the classification of the entire image runs faster.

# Approach:

1) Import necessary modules, libraries such as RandomNormal, tqdm, keras, tensorflow.keras.optimizers Adam
 
2) Here also we had used the metrics such as f1 score with value: 0.589, recall: 0.547, precision: 0.638

3) Define our loss as MSE: 18.043, activation: relu, GANs, discrimanator as reqiuired in model architecture.

4) Read images, labels

5) Define model for discriminator, generator, GAN

6) Training start: epoch: 4, batch_size: 64 for every 4 step we train discriminator as compared to generator (we train discriminator only for the multiple of 4)

7) Fine-tuning Hyper-Parameters to enhance accuracy of our system

# Architecture: DeepLab V3+

# About the Architecture:

The contribution of DeepLab is the introduction of atrous convolutions, or dilated convolutions, to extract more dense features where information is better preserved given objects of varying scale.

Atruos convolutions have an additional parameter r, called the atrous rate, which corresponds to the stride the input signal is sampled at. 

It is equivalent to inserting r-1 zeros between two consecutive filter values along each spacial dimension. 

In this case, r=2 and therefore the amount of zeros between each filter value is 1. 

Here we tune a value called the output_stride, which is the ratio between the input image resolution and the output resolution. Then compare and combine three methodologies to create their finalized approach: Atrous Spatial Pyramid Pooling (ASPP).

The purpose of this methodology is to have the flexibility to modify the filter’s field-of-view and to modify how dense the features are computed, all by changing r instead of learning extra parameters.

# Why DeepLab V3:

One of the challenges in segmenting objects in images using deep convolutional neural networks (DCNNs) is that as the input feature map grows smaller from traversing through the network, information about objects of a smaller scale can be lost hence DeepLab acts as better alternative to deal with information loss.

# Approach:

1) Import necessary modules, libraries tensorflow, keras, matplotlib

2) Read images and labels in array

3) Define Convolution block and DilatedSpatialPyramidPooling operation

4) Here also we had used the metrics such as f1 score: 0.4115, recall: 0.9569, precision: 0.2791, accuracy: 0.7391

5) DeeplabV3plus pretrained: resnet50, weights: data used for training imagenet

6) categorical cross entropy loss with value of 119.4627 and parameters used are epoch: 25, optimizer: ADAM
