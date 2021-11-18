import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
import cv2
from tqdm.auto import tqdm




def find_mean_img_dimensions(image_path, filenames):
    """Finds the mean width and height from images in a directory. 

        Args: 
            image_path: Path to directory of images.
            filenames: Filenames of the images.
      
        Returns:
            Tuple that contains the mean height and width of all the images.
    """
    total_height = 0
    total_width = 0
    length = len(filenames)
    for file in tqdm(filenames):
      image = cv2.imread(image_path + file)
      height, width = image.shape[:2]
      total_height += height
      total_width += width
    

    mean_height = total_height / length
    mean_width = total_width / length

    return (mean_height,mean_width)




@tf.function
def load_img_and_mask(image_path, 
                       mask_path,
                       filename):
    """Loads an image and its masks from their respective directories into TF tensors. 

        Args:
          image_path: Path to the image directory.
          mask_path: Path to the mask directory.
          filename: The filename of the image and mask to load. 
              - Note: images and corresponding masks have identical names. 

        Returns:
          Tuple containing image and mask as uint8 tensors.
    """
    #read image 
    image = tf.io.read_file(image_path + filename)
    #decode jpeg into tensor
    image = tf.image.decode_jpeg(image, channels=3)

    mask_filename = (mask_path + filename)
    #read mask
    mask = tf.io.read_file(mask_filename)
    #decode mask into tensor
    mask = tf.image.decode_image(mask, channels=1, expand_animations = False)

    return (image, mask)




@tf.function
def split_dataset(dataset, dataset_size, 
                train_split=0.8, val_split=0.1, test_split=0.1, 
                shuffle=True, shuffle_size=10000, seed=42):
    """Splits a TF dataset into train, validation, and test sets. 

        Args: 
            dataset: The dataset to split.
            dataset_size: Size of the datset.
            train_split: Training proportion.
            val_split: Validation proportion.
            test_split: Test proportion.
            shuffle: Boolean for if dataset should be shuffled.
            shuffle_size: Size to use for shuffling. 
            seed: Random seed used in shuffle. 
        
        Returns:
            Three TF Datasets corresponding to each set.

    """
    assert (train_split + val_split + test_split) == 1

    if shuffle:
        dataset = dataset.shuffle(shuffle_size, seed)

    #determine the size of each set
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = int(test_split * dataset_size)

    #generate sets
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    return train_dataset, val_dataset, test_dataset




@tf.function
def scale_values(image,mask,mask_split_threshold = 128):
    """Scales image to fit within the scale [0,1] 
    and ensures mask is applied properly by binarizing
    based on the passed in threshold . 

        Args:
            images: The images to scale.
            masks: The masks to encode.
            mask_split_threshold
        
        Returns: 
            A tuple containg the images within the scale [0,1]
            and the mask which is binarized based on 
            the threshold.
    """
    image = tf.math.divide(image, 255)
    mask = tf.where(mask > mask_split_threshold, 1, 0)
    return (image, mask)




@tf.function
def resize_and_pad(image, mask, target_height=512, target_width=512):
    """This function resizes and pads an image
    and its mask to the target height and width.
    
        Args:
            image: The image to process.
            mask: The mask to process.
            target_height: The height to resize to.
            target_width: The width to resize to. 
        
        Returns:
            A tuple containing the image and mask after 
            resizing and padding.
    """
    image = tf.image.resize_with_pad(image,
                               target_height,
                               target_width,
                               method=tf.image.ResizeMethod.BILINEAR,
                               antialias=False)
    mask = tf.image.resize_with_pad(mask,
                               target_height,
                               target_width,
                               method=tf.image.ResizeMethod.BILINEAR,
                               antialias=False)
    return (image, mask)




def display_img_mask_tensors(image_mask_pair):
    """Displays an image and its mask that are stored at TF tensors.

        Args: 
            image_mask_pair: A tuple containing the image and mask to display.

        Returns:
            None
    """
    (image,mask) = image_mask_pair
    #convert to np array
    image_np = np.array(image)
    #squeeze mask to remove dimension with size 1
    mask_np = np.squeeze(mask)
    fig, ax = plt.subplots(1, 2, figsize=(10, 2*10), constrained_layout=True)
    ax[0].imshow(image_np)
    ax[1].imshow(mask_np)