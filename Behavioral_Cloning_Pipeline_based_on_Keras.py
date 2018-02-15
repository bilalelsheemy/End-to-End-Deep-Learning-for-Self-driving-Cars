
# coding: utf-8

# # Behavioral_Cloning_Pipeline_based_on_Keras

# This pipeline contains the following steps:
# 
# * Reading & Loading the Collected Data: my dataset is based on (1- Udacity collected dataset 2- Recovery data helps the model to get back to the center of the lane when it gets into the road ledges).
# 
# * Data Preprocessing: consists of three main steps (1- Cropping the top and bottom redundant segments of each image as they alwas contain unuseful data. 2- Resizing each image to suite the model expected input shape (66,200,3)   3- Converting the image color space into YUV space as Nvidia paper recommends this.)
# 
# * Data Augmentation: consists of two main steps (1- Data Flipping. 2- Random Shadding).
# 
# * Data Visulaizing: Exploring random samples of the preprocessed data and assuring the dataset balancing.
# 
# * Data Batching: Python generators provide us with a very powerful utility that we can load a specified batch of data only when we're in need for it, (i.e so no need to load all the data set at the same time as this is most likely will not fit into our memory so we just load a specified batch on a fly!)
# 
# * Model Training: I've implemented the architecture mentioned in this paper.
# 
# * Model Validation: 20% of the collected data to be considered as validation set.
# 
# * Visualizing both of training & validation loss to make sure that the model doesn't suffer from overfitting or underfitting.
# 

# In[2]:


## Importing the necessary libs

import numpy as np
import pandas as pd
import csv
import cv2
import sklearn
import random, pylab
import matplotlib
import matplotlib.image as mpimg
#import keras
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img

from skimage.exposure import equalize_adapthist
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, axis
get_ipython().run_line_magic('matplotlib', 'inline')

import os, sys
from random import shuffle


# # Data Preprocessing & Augmentation APIs

# In[3]:


## Resizing API
def img_resize(img):
    
    resized_img = cv2.resize(img, (img_width,img_height))
    return resized_img

## Cropping API
def img_crop(img):
    cropped_img = img[50:140,:,:] ## This should trim the top 50 rows and the bottom 20 rows of each image.
    return cropped_img

## Changing Colorspace
def YUV_convert(img):
    YUV_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) ## 
    return YUV_img

def random_shadow(img):
    """
    Applying dark shadow to random segments of the image.
        Parameters:
            image: The input image.
    """
    if(np.random.uniform(0,1) > 0.4): ## This should randomly shade only 50% of data
        
        img_height, img_width = img.shape[0], img.shape[1]
        [x1, x2] = np.random.choice(img_width, 2, replace=False)
        k = img_height / (x2 - x1)
        b = - k * x1
        for i in range(img_height):
            c = int((i - b) / k)
            img[i, :c, :] = (img[i, :c, :] * .5).astype(np.int32)
    return img

def horizontal_flip(img, steer):
    """
    Flipping the images horizontally to aaugment the curved images.
        Parameters:
            image: The input image.
    """
    
    vertical_img = cv2.flip( img, 1 ) ## This should produce mirrored images
    steer = -steer
    return vertical_img, steer

def random_brightness(image):
    """
    Altering the brightness of the input image.
        Parameters:
            image: The input image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def random_transform(image, steering_angle):
    """
    Applying random translation to the image.
        Parameters:
            image: The input image.
    """
    range_x = 100
    range_y = 10
    if(np.random.uniform(0,1) > 0.5):
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        steering_angle += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))    
    #if(np.random.uniform(0,1) > 0.2):
    #    image = random_brightness(image)    
    return image, steering_angle


# In[4]:


## Sets the plts font

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)


# In[5]:


"""
This processing is applied to each frame before passing it to the model either for taining or testing

Firstly, I've used grayscale image with CLAHE Histogram for contrast enhancement firstly but it showed poor performance 
on the data, then ended up using YUV color space which showed up a very stable & correct steers along the track. 
"""
def image_preprocess(img):
    
    cropped_img = img_crop(img)
    resized_img = img_resize(cropped_img)
    YUV_img = YUV_convert(resized_img)
    #gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    #gray = np.asarray(gray)
    #eulized_img = equalize_adapthist(gray)
    #eulized_img = np.reshape(np.ravel(eulized_img), (66, 200, 1))
    return YUV_img


# In[6]:


"""
Displaying random samples of any data set with their corresponding lables

Parameters: Images dataset , Labels as file names.
"""
def display_DataSample(dataSet, file_names):
    fig = plt.figure(figsize=(50, 50))  
    for i in range(10): 
        index = random.randint(0, (len(dataSet)-1))
        image = dataSet[index].squeeze()
        
        sub = fig.add_subplot(5, 2, i+1)
        sub.imshow(image, interpolation='nearest')
        plt.title(file_names[index])
       
    return


# # Data Extractor API

# * This API is responsible for extracting the center,left and right images with their names and their corresponsing steering angles for any given random driving log using randomData_Display() wrapper.
# 
# * I've implemented this API to help me extracting any data in order to preprocess them or visualize any related statistics of them (i.e. This API is almost similar to the generator used in the training below but I've used it as an independent utility to extract data randomly).
# 
# * Also it contains the data augmentation steps which are (1- Flipping all the images with non-zero steering angles and augment the data by the flipped images (i.e. Because flipping images with zero steering angles doesn't seem to help.) 2- Randomly Shadding 50% of the collected images and augment the data by the shaded images.)

# In[13]:


## Data Extractor API
"""
This function acts as a generator but only for extarcting data for exploration and visualization not for training.
Parameters: Random Data Patch (Images & Steers &Labels)
"""
def data_extractor(random_batch):
    
    car_images=[]
    steering_angles=[]
    file_names = []  
    
    for batch_sample in random_batch:
                    
                    
                   
        ### Center Camera Data
        batch_sample_center = batch_sample[0]
        redund_path_center, filename_center = os.path.split(batch_sample_center)
        
        current_path_center = 'Udacitydata/data/IMG/' + filename_center
        image_center = mpimg.imread(current_path_center)
        
        steering_center = float(batch_sample[3])
        correction = 0.2
        
        if image_center is not None:
            """
            Augmenting our data by the left & right camera images don't produce useful data when the
            steering angle is zero.
            """
            if(steering_center == 0):
                if(np.random.uniform(0,1) > 0.95): #Dataset Balancing
                    car_images.append(image_center)
                    steering_angles.append(steering_center)
                    file_names.append(filename_center)
                    
                    shaded_img_center = random_shadow(image_center)
                    car_images.append(shaded_img_center)
                    steering_angles.append(steering_center)
                    file_names.append(filename_center+' "shaded"')
            
            
        
            
                
            if(steering_center != 0):
                car_images.append(image_center)
                steering_angles.append(steering_center)
                file_names.append(filename_center)
                
                flipped_image_center, flipped_steering_center = horizontal_flip(image_center, steering_center)
            
                
                car_images.append(flipped_image_center)
                steering_angles.append(flipped_steering_center)
                file_names.append(filename_center+' "flipped"')
                
                shaded_img_center = random_shadow(image_center)
                car_images.append(shaded_img_center)
                steering_angles.append(steering_center)
                file_names.append(filename_center+' "shaded"')
                
                #bright_img_center, steering_center_center = random_transform(image_center, steering_center)
                #car_images.append(bright_img_center)
                #steering_angles.append(steering_center_center)
        
        
        if(steering_center != 0):
        
            ### Left Camera Data
            batch_sample_left = batch_sample[1]
            redund_path_left, filename_left = os.path.split(batch_sample_left)
            current_path_left = 'Udacitydata/data/IMG/' + filename_left
            image_left = mpimg.imread(current_path_left)
            
            
            # create adjusted steering measurements for the side camera images
            # this is a parameter to tune
            steering_left = steering_center + correction
            
            if image_left is not None:
                car_images.append(image_left)
                steering_angles.append(steering_left)
                file_names.append(filename_left)
                
                shaded_img_left = random_shadow(image_left)
                car_images.append(shaded_img_left)
                steering_angles.append(steering_left)
                file_names.append(filename_left+' "shaded"')
                
                flipped_image_left, flipped_steering_left = horizontal_flip(image_left, steering_left)
                    
                car_images.append(flipped_image_left)
                steering_angles.append(flipped_steering_left)
                file_names.append(filename_left+' "flipped"')
                
                #bright_img_left, steering_left_left = random_transform(image_left, steering_left)
                #car_images.append(bright_img_left)
                #steering_angles.append(steering_left_left)
                
            ### Right Camera Data         
            batch_sample_right = batch_sample[2]
            redund_path_right, filename_right = os.path.split(batch_sample_right)
            current_path_right = 'Udacitydata/data/IMG/' + filename_right
            image_right = mpimg.imread(current_path_right)
            
            
            # create adjusted steering measurements for the side camera images
            # this is a parameter to tune
            steering_right = steering_center - correction
        
            if image_right is not None:
                car_images.append(image_right)
                steering_angles.append(steering_right)
                file_names.append(filename_right)
                
                shaded_img_right = random_shadow(image_right)
                car_images.append(shaded_img_right)
                steering_angles.append(steering_right)
                file_names.append(filename_right+' "shaded"')
                
                flipped_image_right, flipped_steering_right = horizontal_flip(image_right, steering_right)
                 
                    
                car_images.append(flipped_image_right)
                steering_angles.append(flipped_steering_right)
                file_names.append(filename_right+' "flipped"')
                        
                    
                            
                        
        
        
    return car_images, steering_angles, file_names


# # Displaying Random Data From a Given Batch

# This API acts as a wrapper abstracting the user from data_extractor() API directly, as it randomly selects random samples from the given data batch and then calls data_extractor() API to handle the useful data extraction then calls display_DataSample() to display the images with their corresponding labels.
# 
# 
# <img src="behClon.png">
# 
# 

# In[8]:


"""
This API acts as warapper which calls data_extractor() to load the data and then display them using display_DataSample()
Parameters: Data Patch (Images & Steers &Labels).
"""
def randomData_Display(sampleData):
    
    random_batch=[]
    
    for smapleCntr in range(100): ## 100 is adjustable range for the random samples to be selected
        index = random.randint(0, len(sampleData))
        random_batch.append(sampleData[index])
    
    img_list, steers_list, file_names = data_extractor(random_batch) ## Data Extraction
    
    img_list_prep = list(map(image_preprocess, img_list)) ## In order to visualize the preprocessing results
    display_DataSample(img_list_prep, file_names)

            


# # Dataset Balance Explorer

# This API explores the representation of each steering value in the dataset (Note that: we've taken steps to balance the dataset in the data_extractor() API).

# In[9]:


"""
This API diplays the histogram of the passed data set in order to explore the data set balancing.
You'll observe that the both training and validation data sets are fairly balanced here as I've taken steps to balance them
through down sampling the zero steers images representations in both datasets.
"""
def dataset_distribution(datasample):
    
    img_list, steers_list, file_name = data_extractor(datasample)

    fig=plt.figure()
    dataset = fig.add_subplot(1,1,1)
    
    #Axes Data
    dataset.hist( steers_list ,bins = 30, rwidth = 0.6) 
    
    #Labels and Tit
    plt.title('Steering Angles Distribution')
    plt.xlabel('Steering Angel Value')
    plt.ylabel('Representation in the Data set')
    plt.show()
    


# # Reading & Loading the Collected Data

# # Data Visulaizing & Exploring Dataset Balance

# In[11]:


from sklearn.model_selection import train_test_split

lines = []

with open('Udacitydata/data/driving_log.csv') as csvfile: 
    training_readings = csv.reader(csvfile)


    for line in training_readings:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


# In[36]:


randomData_Display(train_samples)


# In[38]:


randomData_Display(validation_samples)


# # Dataset Before Balancing

# In[12]:


"""
The original data set histogram
"""
dataset_distribution(train_samples)


# # Dataset After Balancing

# In[14]:


"""
The balanced data set histogram
"""
dataset_distribution(train_samples)


# 
# # Data Preprocessing & Augmentation Using Generator

# Generator acts as batcher to continously supply the needed data on time without loading the full dataset.
# 
# * For the training purposes the batcher will consider the left & right camera data also it'll preprocess and augment the whole data through flipping and shading.
# 
# * For the the validation purposes the batcher will consider the center camera data only and will not apply any augmentation, only it'll preprocess the input images.

# In[12]:


def generator(samples, batch_size=32, training_flag=1):
    
    if(training_flag == 1):
        
        num_samples = len(samples)
        while True: # Loop forever so the generator never terminates
                    shuffle(samples)
                    for offset in range(0, num_samples, batch_size):
                        batch_samples = samples[offset:offset+batch_size]

                        car_images=[]
                        steering_angles=[]
                        file_names = [] 
                        
                        for batch_sample in batch_samples:
                    
                    
                   
                            ### Center Camera Data
                            batch_sample_center = batch_sample[0]
                            redund_path_center, filename_center = os.path.split(batch_sample_center)
                            
                            current_path_center = 'Udacitydata/data/IMG/' + filename_center
                            image_center = mpimg.imread(current_path_center)
                    
                            steering_center = float(batch_sample[3])
                            correction = 0.2
                            
                            if image_center is not None:
                                """
                                Augmenting our data by the left & right camera images don't produce useful data when the
                                steering angle is zero.
                                """
                                if(steering_center == 0):
                                    if(np.random.uniform(0,1) > 0.95): #Dataset Balancing
                                        car_images.append(image_center)
                                        steering_angles.append(steering_center)
                                        file_names.append(filename_center)
                                        
                                        shaded_img_center = random_shadow(image_center)
                                        car_images.append(shaded_img_center)
                                        steering_angles.append(steering_center)
                                        file_names.append(filename_center+' "shaded"')
                                
                                
                            
                                
                                    
                                if(steering_center != 0):
                                    car_images.append(image_center)
                                    steering_angles.append(steering_center)
                                    file_names.append(filename_center)
                                    
                                    flipped_image_center, flipped_steering_center = horizontal_flip(image_center, steering_center)
                                    
                                    car_images.append(flipped_image_center)
                                    steering_angles.append(flipped_steering_center)
                                    file_names.append(filename_center+' "flipped"')
                                    
                                    shaded_img_center = random_shadow(image_center)
                                    car_images.append(shaded_img_center)
                                    steering_angles.append(steering_center)
                                    file_names.append(filename_center+' "shaded"')
                                    
                                    #bright_img_center, steering_center_center = random_transform(image_center, steering_center)
                                    #car_images.append(bright_img_center)
                                    #steering_angles.append(steering_center_center)
                        
                         
                            if(steering_center != 0):
                            
                                ### Left Camera Data
                                batch_sample_left = batch_sample[1]
                                redund_path_left, filename_left = os.path.split(batch_sample_left)
                                current_path_left = 'Udacitydata/data/IMG/' + filename_left
                                image_left = mpimg.imread(current_path_left)
                                
                                
                                # create adjusted steering measurements for the side camera images
                                # this is a parameter to tune
                                steering_left = steering_center + correction
                                
                                if image_left is not None:
                                    car_images.append(image_left)
                                    steering_angles.append(steering_left)
                                    file_names.append(filename_left)
                                    
                                    shaded_img_left = random_shadow(image_left)
                                    car_images.append(shaded_img_left)
                                    steering_angles.append(steering_left)
                                    file_names.append(filename_left+' "shaded"')
                                    
                                    flipped_image_left, flipped_steering_left = horizontal_flip(image_left, steering_left)
                                        
                                    car_images.append(flipped_image_left)
                                    steering_angles.append(flipped_steering_left)
                                    file_names.append(filename_left+' "flipped"')
                                    
                                    #bright_img_left, steering_left_left = random_transform(image_left, steering_left)
                                    #car_images.append(bright_img_left)
                                    #steering_angles.append(steering_left_left)
                                    
                                ### Right Camera Data         
                                batch_sample_right = batch_sample[2]
                                redund_path_right, filename_right = os.path.split(batch_sample_right)
                                current_path_right = 'Udacitydata/data/IMG/' + filename_right
                                image_right = mpimg.imread(current_path_right)
                                
                                
                                # create adjusted steering measurements for the side camera images
                                # this is a parameter to tune
                                steering_right = steering_center - correction
                
                                if image_right is not None:
                                    car_images.append(image_right)
                                    steering_angles.append(steering_right)
                                    file_names.append(filename_right)
                                    
                                    shaded_img_right = random_shadow(image_right)
                                    car_images.append(shaded_img_right)
                                    steering_angles.append(steering_right)
                                    file_names.append(filename_right+' "shaded"')
                                    
                                    flipped_image_right, flipped_steering_right = horizontal_flip(image_right, steering_right)
                                        
                                    car_images.append(flipped_image_right)
                                    steering_angles.append(flipped_steering_right)
                                    file_names.append(filename_right+' "flipped"')
                                    
                                    #bright_img_right, steering_right_right = random_transform(image_right, steering_right)
                                    #car_images.append(bright_img_right)
                                    #steering_angles.append(steering_right_right)
                        
                        
                    
                        ## All the images and corresponding steering angles collected
        
                    
                        ## Resizing & Cropping step
                        xtrain_preplist = []
                        
                        
                        xtrain_preplist = list(map(image_preprocess, car_images))
                        xtrain_prep = np.asarray(xtrain_preplist)
                        
                        
                        x_train = np.asarray(xtrain_prep)
                        y_train = np.asarray(steering_angles) 
                        yield sklearn.utils.shuffle(x_train, y_train) 
                        
    else:
        
        num_samples = len(samples)
        while True: # Loop forever so the generator never terminates
                shuffle(samples)
                for offset in range(0, num_samples, batch_size):
                    batch_samples = samples[offset:offset+batch_size]

                    car_images=[]
                    steering_angles=[]
                        
                    for batch_sample in batch_samples:
                        
                         ### Center Camera Data
                        batch_sample_center = batch_sample[0]
                        redund_path_center, filename_center = os.path.split(batch_sample_center)
                        current_path_center = 'Udacitydata/data/IMG/' + filename_center
                        image_center = mpimg.imread(current_path_center)
                    
                        steering_center = float(batch_sample[3])
                        
                        
                        if image_center is not None:
                            car_images.append(image_center)
                            steering_angles.append(steering_center)
                            
                        
                    xvalid_preplist = []
                    
                    
                    xvalid_preplist = list(map(image_preprocess, car_images))
                    xvalid_prep = np.asarray(xvalid_preplist)
    
                    
                    x_valid = np.asarray(xvalid_prep)
                    y_valid = np.asarray(steering_angles) 
                    yield sklearn.utils.shuffle(x_valid, y_valid)

    


# # Model Training & Validating

# As per the network implemented in this paper >> 
# 
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
# 
# * I will not demonstrate the network architecture as it's well demonstrated in the referred link above.
# 
# * I've implemented their network including the Lambda layer they mentioned for normalizing purposes.
# 
# * I've used Adam optimizer which is an extension of the stochastic gradient descent for this problem with learning rate = 1.0e-4. 
# 
# * I've used MSE as a loss function as this is not a classification problem, it's a regression one.
# 
# * For more info why Adam optimizer was selected >> 
# 
# * AdamOptimizer is combining the advantages of two other extensions of stochastic gradient descent. Specifically:
# 
# * Adaptive Gradient Algorithm (AdaGrad) that maintains a per-parameter learning rate that improves performance on problems with sparse gradients (e.g. natural language and computer vision problems).
# 
# * Root Mean Square Propagation (RMSProp) that also maintains per-parameter learning rates that are adapted based on the average of recent magnitudes of the gradients for the weight (e.g. how quickly it is changing). This means the algorithm does well on online and non-stationary problems (e.g. noisy).
# 
# * Adam was demonstrated empirically to show that convergence meets the expectations of the theoretical analysis. Adam was applied to the logistic regression algorithm on the MNIST character recognition and IMDB sentiment analysis datasets, a Multilayer Perceptron algorithm on the MNIST dataset and Convolutional Neural Networks on the CIFAR-10 image recognition dataset.
# 
# And here's Comparison of Adam to Other Optimization Algorithms Training a Multilayer Perceptron
# * Taken from Adam: A Method for Stochastic Optimization, 2015.
# 
# <img src="Comparison-of-Adam-to-Other-Optimization-Algorithms-Training-a-Multilayer-Perceptron.png">
# 
# 
# 
# * As mentioned above training data is based on the Udacity data and recovery data I've collected through putting the car near or towards the road ledges then recording while I'm take it back to the center of the road. Because Udacity dataset is biased to drive straight most of the time (i.e. steering = 0) and this noticeably raised while balancing the dataset, so the recovery data helped in improving model performance when car is not in the road center.
# 

# # Model Architecture

# In[24]:


train_generator = generator(train_samples, batch_size=32, training_flag=1)
validation_generator = generator(validation_samples, batch_size=32, training_flag=0)


# In[29]:


from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Reshape, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
img_height = 66
img_width = 200
channels = 3
INPUT_SHAPE = img_height, img_width, channels
batch_size=32
model = Sequential()

# set up lambda layer
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=INPUT_SHAPE))

# compile and train the model using the generator function
model.add(Convolution2D(24, 5, 5, subsample=(2, 2),  activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2),  activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2),  activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))

model.add(Flatten(input_shape=(1, 18, 64)))

model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation = 'relu'))
#model.add(Dropout(0.2))
model.add(Dense(10, activation = 'relu'))

model.add(Dense(1))

"""
ADAM Optimizer is used with learning rate = 1.0e-4

"""
model.compile(loss='mse', optimizer=Adam(lr=1.0e-4))

callbacks=[
    #EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto'),
    ModelCheckpoint('model-{val_loss:03f}.h5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             mode='auto'),
]


history_object = model.fit_generator(train_generator, samples_per_epoch=             20000, validation_data=validation_generator,             nb_val_samples=len(validation_samples), nb_epoch=5, callbacks=callbacks,  verbose=1)

model.summary()


# # Visulaizing Training & Validation Losses

# In[32]:


"""
Visulaizing the training and validation loss for each epoch
This helps to observe if the model underfit or overfit the data set.

"""
print(history_object.history.keys())

fig = pylab.figure(figsize=(12,9))

loss_visulaization = fig.add_subplot(211)
loss_visulaization.plot(history_object.history['loss'])
loss_visulaization.plot(history_object.history['val_loss'])

loss_visulaization.set_ylim([0,0.05])
loss_visulaization.set_xlim([0,7])

plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


