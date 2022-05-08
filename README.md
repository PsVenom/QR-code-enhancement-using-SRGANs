# QR code enhancer
An attempt to ustilise Super Resolution Generative Adversarial Networks (SRGANs) on QR codes to enhance images.  

Most qr scanners require the scanning image to be greater than 2cms x 2cms. However, I believe we can shrink the size by a respectable factor. This can be made possibile by artificial resolution-enhancement. This application opens a lot of possibilities for significantly quicker realtime detection 
## References
Link to dataset- https://www.kaggle.com/datasets/coledie/qr-codes

Link to the original paper- https://arxiv.org/abs/1609.04802

Excellent repo I based my work on - https://github.com/bnsreenu/python_for_microscopists/tree/master/255_256_SRGAN
# Files in repository
main.py ---> contains the main model architecture

Superresolution.ipynp ---> A detailed notebook for downloading the model, augmenting image data, preparing and compiling the final model, and finally running it
# Understanding SRGAN architecture 
SRGAN is a generative adversarial network for single image super-resolution. It uses a perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes the solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. 

## The Generator

![image](https://user-images.githubusercontent.com/99831413/166132910-9a7e508b-bead-4599-904f-ece3f38a0845.png)

 The generator architecture of the SRRESNET generator network consists of the low-resolution input, which is passed through an initial convolutional layer of 9×9 kernels and 64 feature maps followed by a Parametric ReLU layer. The reason for choosing the Parametric ReLU is because it is one of the best non-linear functions for this particular task of mapping low-resolution images to high-resolution images.
 
 ## The Discriminator
 
 ![image](https://user-images.githubusercontent.com/99831413/166132890-ee9647be-ec3f-462d-a4ef-517be8d778dd.png)

 
 The discriminator architecture is constructed in the best way to support a typical GAN procedure. Both the generator and discriminator are competing with each other, and they are both improving simultaneously. While the discriminator network tries to find the fake images, the generator tries to produce realistic images so that it can escape the detection from the discriminator. The working in the case of SRGANs is similar as well, where the generative model G with the goal of fooling a differentiable discriminator D that is trained to distinguish super-resolved images from real images.

Hence the discriminator architecture shown in the above image works to differentiate between the super-resolution images and the real images.


# Performance so far

## Epoch one

![download](https://user-images.githubusercontent.com/99831413/166133210-64a050a4-79f4-4d19-b931-96d0e80ad335.png)

## Epoch two

![image](https://user-images.githubusercontent.com/99831413/166930498-8cbf29d5-d2d0-454a-b084-1add23e8d545.png)

## Epoch three

![image](https://user-images.githubusercontent.com/99831413/167305009-329cca3b-c2a0-40b6-954b-083635bafecd.png)

![image](https://user-images.githubusercontent.com/99831413/167304994-ca4238de-ef66-4ca0-9d98-b05bf6abe320.png)

### we can safely say that the results are nearly identical to the actual image
