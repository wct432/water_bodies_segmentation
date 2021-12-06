- [1. Introduction](#1-introduction)
- [2. Exploring Data](#2-exploring-data)
    - [1. Find Number of Images](#1-find-number-of-images)
    - [2. Find Mean Image Dimensions](#2-find-mean-image-dimensions)
    - [3. Load and Display Images as Tensors](#3-load-and-display-images-as-tensors)
        - [1. Define Function to Load Images and Masks](#1-define-function-to-load-images-and-masks)
        - [2. Display Example Images](#2-display-example-images)
- [3. Prepare Data](#3-prepare-data)
    - [1. Create TF Dataset, Scale, and Resize Images](#1-create-tf-dataset-scale-and-resize-images)
    - [2. Display Example after Processing](#2-display-example-after-processing)
    - [3. Create Train, Test, and Validation Sets](#3-create-train-test-and-validation-sets)
    - [4. View Shape of Datasets](#4-view-shape-of-datasets)
- [4. Build and Train U-Net Model](#4-build-and-train-u-net-model)
    - [1. Build U-Net Model](#1-build-u-net-model)
    - [2. Define Model Callbacks](#2-define-model-callbacks)
    - [3. Fit Model](#3-fit-model)
- [5. Evaluate Model Performance](#5-evaluate-model-performance)

# 1. Introduction
In this project we will be performing image segmentation to segment bodies of water in satellite images using a Convolutional Neural Network based on the U-Net architecture. 

Our dataset consists of images captured via satellite and our model's task will be to segment bodies of water that are present in the image. 

We will begin by exploring our dataset, loading our jpeg images as tensors in a Tensorflow dataset, and then then scaling the images and resizing them based on the mean image height and width of our dataset. 

The U-Net architecture our model is based on is known for its U shape. It consists of two main paths, the left path is known as the contracting path, or encoder, and is a three tiered convolutional network that captures the image, the right path is referred to as the expansive path, or decoder, and has three upsampling convolutional layers that expand the low resolution features into a high resolution tensor. The two paths are joined by two central convolution layers.

I based my model on U-Net because it exceedes at image segmentation tasks, and as we will see it performs well for our useage. 

# 2. Exploring Data
First we will begin by exploring our data and viewing the images and the masks. 

### 1. Find Number of Images
First we will just find the number of images in the dataset:
``` python
#print number of images in our dataset
num_images = len(filenames)
print(f'Number of Images: {num_images}')
```

### 2. Find Mean Image Dimensions
Next we will find the mean height and width of our image. Before passing images to our model they will have to be resized so they all have uniform dimensions, so we want to find the average dimensions in order to guide our resizing.    

I defined a function in our utils module called find_mean_img_dimensions that will find the mean height and width of all the images in a specified file path using Open CV. Here is the function definition:
``` python
def find_mean_img_dimensions(image_path, filenames):
    """Finds the mean width and height from images in a directory. 

        Args: 
            image_path: Path to directory of images.
            filenames: Filenames of the images.
      
        Returns:
            Tuple that contains the mean height and width of all the images.
    """
    total_height = 0                            #track total height
    total_width = 0                             #track total width
    length = len(filenames)                     #number of files in directory
    for file in tqdm(filenames):                #for each file in directory
        image = cv2.imread(image_path + file)   #load image using Open CV
        height, width = image.shape[:2]         #capture image height and width
        total_height += height                  #add height to total height
        total_width += width                    #add width to total width
    

    mean_height = total_height / length         #divide total height by number of images to find mean height
    mean_width = total_width / length           #divide total width by number of images to find mean width

    return (mean_height,mean_width)             #return mean height and width as a tuple
```

Finding mean dimensions:
``` python
# find the mean height and width of our images
(mean_height, mean_width) = find_mean_img_dimensions(IMAGE_PATH,filenames)  

print(f'Mean Height: {mean_height}')
print(f'Mean Width: {mean_width}')
```
*Mean Height: 560.6652587117212  
Mean Width: 445.2555438225977*


### 3. Load and Display Images as Tensors
Next we are going to load our images as Tensorflow tensors and then displaying them using MatPlotLib. To load the images as tensors for display as well as to load them into our dataset I wrote a function that takes a filepath for the image, mask, and a filename and loads images and their corresponding masks as tensors. Note that image and masks must have identical file names, and only seperated by their directory.

##### 1. Define Function to Load Images and Masks
``` python
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
    image = tf.io.read_file(image_path + filename)              #read file into buffer                    
    image = tf.image.decode_jpeg(image, channels=3)             #decode jpeg into tensor
    mask_filename = (mask_path + filename)                      #create full filename for masks
    mask = tf.io.read_file(mask_filename)                       #read masks
    mask = tf.image.decode_image(mask, channels=1)              #decode mask into tensor

    return (image, mask)                                        #return image and mask as a tuple
```

##### 2. Display Example Images 
Next we will display three sample images and masks from our dataset, first we will group three random examples together: 
``` python
#load three random examples
n_examples = 3
examples = [load_img_and_mask(IMAGE_PATH,
                              MASK_PATH,
                              filenames[random.randrange(len(filenames))]) \
            for i in range(n_examples)]
```
Plotting Example Images:
``` python
#plot our examples
fig, axes = plt.subplots(n_examples, 2, figsize=(14, n_examples*7), constrained_layout=True)
for ax, (image, mask) in zip(axes, examples):
    mask = tf.squeeze(mask)
    ax[0].imshow(image)
    ax[1].imshow(mask)
```
![Example Images](/../images/images/image_mask_samples.png?raw=true)

# 3. Prepare Data
Now that we have an overview of the data we're working with we are ready to create a Tensorflow Dataset for training, preprocess and transform the images, and prepare train, test, and validation sets for our model.   

We will:
    <ul>
        <li>create a Tensorflow dataset for our images and masks</li>
        <li>reshape our images to have uniform dimensions based on the mean dimensions found earlier</li>
        <li>standardize our RGB values to fit in the [0,1] range</li>
        <li>create train, test, and validation sets</li>
    </ul>


### 1. Create TF Dataset, Scale, and Resize Images
``` python
RANDOM_STATE = 42 #used to split dataset into train,val,test
TARGET_HEIGHT = 512 #target height to resize images to
TARGET_WIDTH = 512 #target width to resize images to

#create TF dataset from filenames
dataset = tf.data.Dataset.from_tensor_slices(filenames)

#use our load_img_and_mask function to load each image and mask into the dataset
dataset = dataset.map(lambda x: load_img_and_mask(IMAGE_PATH,
                                                  MASK_PATH,
                                                  filename=x))

#scale the images to fit within the scale [0,1] and replaces any values that are not binary in the masks with 1 or 0 based on the threshold of 128
dataset = dataset.map(scale_values)

#resizes and pads the images to the target height and target width that we found earlier 
dataset = dataset.map(lambda image, mask: resize_and_pad(image, mask, target_height=TARGET_HEIGHT,target_width=TARGET_WIDTH))
```

Now we have created a Tensorflow Dataset to hold our data, which has also been scaled and resized and padded so that all images have a uniform dimension.

### 2. Display Example after Processing
Now we will display three example images to show how they look after processing:
``` python
#display three images from the dataset to see what they look like after processing
n_examples = 3
for samples in dataset.take(n_examples):
  display_img_mask_tensors(samples)
```
![Example Image 1 after Processing](/../images/images/post_processing_1.png?raw=true)
![Example Image 2 after Processing](/../images/images/post_processing_2.png?raw=true)
![Example Image 3 after Processing](/../images/images/post_processing_3.png?raw=true)

### 3. Create Train, Test, and Validation Sets
We will now create Train, Test, and Validation sets using the split_dataset function defined in utils. We will also prefetch 1 batch of 5 samples at a time to speed up training. This allows elements to be prepared while current elements are being processed, improving latency at the cost of additional memory.

``` python
 #split dataset into train test and validation sets
train_dataset,val_dataset,test_dataset = split_dataset(dataset, num_images, 
                                                       shuffle_size=num_images,
                                                       seed=RANDOM_STATE)
train_dataset = train_dataset.batch(1).prefetch(5) #prefetch 1 batch of 5 samples 
val_dataset = val_dataset.batch(1).prefetch(5)
test_dataset = test_dataset.batch(1).prefetch(5)
```


### 4. View Shape of Datasets
Next we will view the shapes for each dataset to ensure everything looks correct. 
``` python
#ensure the dataset shapes look correct
print(f'Original Set: {dataset}')
print(f'Training Set: {train_dataset}')
print(f'Validation Set: {val_dataset}')
print(f'Testing Set: {test_dataset}')
```
*Original Set: <MapDataset shapes: ((512, 512, 3), (512, 512, 1)), types: (tf.float32, tf.float32)>  
Training Set: <PrefetchDataset shapes: ((None, 512, 512, 3), (None, 512, 512, 1)), types: (tf.float32, tf.float32)>  
Validation Set: <PrefetchDataset shapes: ((None, 512, 512, 3), (None, 512, 512, 1)), types: (tf.float32, tf.float32)>  
Testing Set: <PrefetchDataset shapes: ((None, 512, 512, 3), (None, 512, 512, 1)), types: (tf.float32, tf.float32)>*  



# 4. Build and Train U-Net Model
Now we are ready to build our model using Kera's functional API and the get_unet function defined in our build_model module.  
This function will build a CNN based on the U-Net architecture.   

As mentioned before the left path of the model is known as the contracting path, or encoder, and is a three tiered convolutional network that captures the image using three stacks consisting of two Conv2D layers and a max pooling layer, the right path or decoder has three upsampling convolutional layers formed by an UpSampling2D layer and Conv2D layers, which are responsible for expandindg the low resolution features into a high resolution tensor. The two paths are joined by two central Conv2D layers.  

Our model will train for 100 epochs or until early stopping is implemented, and we will save the model that performs best on validation data. Furthermore, we will reduce the learning rate by half when the model's validation loss doesn't improve for two epochs.   

We will use the Nadam optimizer with the default learning rate, which is a variant of the popular Adam optimizer with Nesterov accelerated momentum.   

We will monitor loss and accuracy as the model trains, when it is done training we will measure the Mean Intersection over Union or IoU, on test data, which is a common measure of performance for segmentation models.

### 1. Build U-Net Model
First we will build our model using the get_unet function defined in the build_model module.
This is the function definition:  
``` python
def get_unet(hidden_activation='relu', initializer='he_normal', output_activation='sigmoid'):
    """Builds a Convolutional Neural Network for semantic segmentation 
    with an architecture based on U-Net. 

        Args:
            hidden_activation: The hidden activation function to use, 
            in the convolutional layers, defaults to ReLU.
            initializer: The kernel initializer to use in the convolutional layers,
            he_normal by default.
            output_activation: The output activation function, defaults to sigmoid.

        Returns:
            A Tensorflow Model with two paths, the first path is the encoder
            and is responsible for capturing the context in the image, it is
            a stack of three convolutional layers with Max Pooling.

            The second path forms the decoder and has three upsampling convlutional
            layers to expand the low resolution features from the encoder into a 
            higher resolution for prediction.

            The two paths are connected by two central convolutional layers. 
    """
    PartialConv = partial(Conv2D,
     activation=hidden_activation,
     kernel_initializer=initializer,      
     padding='same')
    
    #ENCODER
    model_input = Input(shape=(None, None, 3))
    enc_cov_1 = PartialConv(32, 3)(model_input)
    enc_cov_1 = PartialConv(32, 3)(enc_cov_1)
    enc_pool_1 = MaxPooling2D(pool_size=(2, 2))(enc_cov_1)
    
    enc_cov_2 = PartialConv(64, 3)(enc_pool_1)
    enc_cov_2 = PartialConv(64, 3)(enc_cov_2)
    enc_pool_2 = MaxPooling2D(pool_size=(2, 2))(enc_cov_2)
    
    enc_cov_3 = PartialConv(128, 3)(enc_pool_2)
    enc_cov_3 = PartialConv(128, 3)(enc_cov_3)
    enc_pool_3 = MaxPooling2D(pool_size=(2, 2))(enc_cov_3)
    
    #CENTER
    center_cov = PartialConv(256, 3)(enc_pool_3)
    center_cov = PartialConv(256, 3)(center_cov)
    
    #DECODER
    upsampling1 = UpSampling2D(size=(2, 2))(center_cov)
    dec_up_conv_1 = PartialConv(128, 2)(upsampling1)
    dec_merged_1 = Concatenate(axis=3)([enc_cov_3, dec_up_conv_1])
    dec_conv_1 = PartialConv(128, 3)(dec_merged_1)
    dec_conv_1 = PartialConv(128, 3)(dec_conv_1)
    
    upsampling2 = UpSampling2D(size=(2, 2))(dec_conv_1)
    dec_up_conv_2 = PartialConv(64, 2)(upsampling2)
    dec_merged_2 = Concatenate(axis=3)([enc_cov_2, dec_up_conv_2])
    dec_conv_2 = PartialConv(64, 3)(dec_merged_2)
    dec_conv_2 = PartialConv(64, 3)(dec_conv_2)
    
    upsampling3 = UpSampling2D(size=(2, 2))(dec_conv_2)
    dec_up_conv_3 = PartialConv(32, 2)(upsampling3)
    dec_merged_3 = Concatenate(axis=3)([enc_cov_1, dec_up_conv_3])
    dec_conv_3 = PartialConv(32, 3)(dec_merged_3)
    dec_conv_3 =  PartialConv(32, 3)(dec_conv_3)
    
    #OUTPUT
    output = Conv2D(1, 1, activation=output_activation)(dec_conv_3)
    
    return tf.keras.Model(inputs=model_input, outputs=output)
```

Now we will create our model using the get_unet function:
``` python
#create model
model = get_unet()
#define optimizer
optimizer = tf.keras.optimizers.Nadam()
#compile model
model.compile(loss='binary_crossentropy', 
              optimizer = optimizer,
              metrics = ["accuracy"])
```

### 2. Define Model Callbacks
Next we will define our callbacks: 
    <ul>
        <li>checkpoint will be used to save the model with the best validation loss</li>
        <li>early_stopping will stop training when the model doesn't improve for 6 epochs</li>
        <li>lr_reduce will reduce the learning rate when the model doesn't improve for 2 epochs</li>
        <li>tqdm_callback simply displays a nice progress bar during training</li>
    </ul>
``` python
checkpoint = ModelCheckpoint(filepath = MODEL_PATH + f'U-NET/model_{dt.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")}_best.h5', 
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=6)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
tqdm_callback = tfa.callbacks.TQDMProgressBar()

callbacks = [checkpoint,
             early_stopping,
             lr_reduce,
             tqdm_callback]
```

### 3. Fit Model
Now we are ready to fit our model and begin training! 
``` python
EPOCHS = 100
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks)
```

# 5. Evaluate Model Performance
Now we will evaluate our model on our test data and measure its Mean Intersection over Union, or IoU.  

IoU a good metric used to measure the overlap between two bounding boxes or masks.  

It measures area of overlap between predicted segmentation and the ground truth divided by the area of union between the predicted segmentation and the ground truth, or: true_positive / (true_positive + false_positive + false_negative).  

A perfect prediction results in an IoU of 1, and the lower IoU the worse the model performance. 

The Mean Intersection-Over-Union first computes the IOU for each semantic class and then computes the average over all classes. Predictions are accumulated in a confusion matrix, weighted by sample_weight and the metric is then calculated from it.  

``` python
meanIoU = tf.keras.metrics.MeanIoU(num_classes=2)   #define meanIoU
test_size = len(test_dataset)                       #find length of test set
for ele in test_dataset.take(test_size):            #for image, mask pair in test set
    image, y_true = ele                             #select image and mask
    prediction = model.predict(image)               #make model prediction based on image
    prediction = tf.where(prediction > 0.5, 1, 0)   #create mask out of prediction
    meanIoU.update_state(y_true, prediction)        #update the state of the meanIoU metric
IoU_result = meanIoU.result().numpy()               #select the Mean IoU score
meanIoU.reset_state()                               #reset state of our mean IoU metric

print(f'Mean IoU: {IoU_result}')                    #print result
```
*Mean IoU: 0.7805627584457397*

Our model generates a mean IoU of 0.78 which tells us that on average it is doing a good job at segmenting the water in images. On average the segmentation our model creates overlaps with the actual water in the image by 0.78%. 