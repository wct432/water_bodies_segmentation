# water_body_segmentation
In this project we will be performing image segmentation to segment bodies of water in satellite images using a Convolutional Neural Network based on the U-Net architecture. 

Our dataset consists of images captured via satellite and our model's task will be to segment bodies of water that are present in the image. 

We will begin by exploring our dataset, loading our jpeg images as tensors in a Tensorflow dataset, and then then scaling the images and resizing them based on the mean image height and width of our dataset. At this point our data is ready to be fed to our model.

The U-Net architecture our model is based on is known for its U shape. It consists of two main paths, the left path is known as the contracting path, or encoder, and is a three tiered convolutional network that captures the image, the right path is referred to as the expansive path, or decoder, and has three upsampling convolutional layers that expand the low resolution features into a high resolution tensor. The two paths are joined by two central convolution layers.

I based my model on U-Net because it exceedes at image segmentation tasks, and as we will see it performs well for our useage. 
