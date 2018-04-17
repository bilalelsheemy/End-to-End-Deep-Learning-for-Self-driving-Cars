# End-to-End-Deep-Learning-for-Self-driving-Cars

## Behavioral_Cloning_Pipeline_Using_Keras

- **Project Goals:**
  - Use the simulator to collect data of good driving behavior.
  - Build, a convolution neural network in Keras that predicts steering angles from images.
  - Train and validate the model with a training and validation set.
  - Test that the model successfully drives around track one without leaving the road.
  - Summarize the results with a written report This pipeline contains the following steps:
  - Reading & Loading the Collected Data: my dataset is based on (1- Udacity collected dataset).
  
  - **Data Preprocessing: consists of three main steps:** 
    - Cropping the top and bottom redundant segments of each image as they always contain unuseful data.
    - Resizing each image to suite the model expected input shape (66,200,3).
    - Converting the image color space into YUV space as Nvidia paper recommends this.
    
  - **Data Augmentation: consists of two main steps** 
    - Data Flipping. 
    - Random Shadding. 
     
  - Data Visulaizing: Exploring random samples of the processed data and assuring the datasets balancing. 
  - Data Batching: Python generators provide us a very powerful utility that we can load a specified batch of data only when we're in need 
  for it, (i.e. so no need to load all the data set at the same time as this is most likely will not fit into our memory so we just load 
  a specified batch on a fly!). 
  - Model Training: I've implemented the architecture mentioned in this paper. 
  - Model Validation: 20% of the collected data to be considered as validation set. 
  - Visualizing both of training & validation loss to make sure that the model doesn't suffer from overfitting or underfitting.
  
##  Rubric Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

- **Files Submitted & Code Quality.**
  - Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:
    - Behavioral_Cloning_Pipeline_based_on_Keras.py containing the script (Pipeline) to create and train the model.
    - drive.py for driving the car in autonomous mode.
    - OutputModel-ValAcc0.011382 containing a trained convolutional neural network.
    - writeup_report.pdf discussing the architecture and concluding the results.
 
- **Submission includes functional code.**
Using the Udacity provided simulator and my drive.py script, the car can be driven autonomously around the track by executing python drive.py OutputModel-ValAcc0.011382.

- **Submission code is usable and readable.**
You’ll find all of the pipeline code and clarifying comments in
Behavioral_Cloning_Pipeline_based_on_Keras.py
Model Architecture and Training Strategy
  - An appropriate model architecture has been employed.
  - Attempts to reduce overfitting in the model.
As you can see I’ve implemented the CNN published in End to End Learning for Self-Driving Cars paper.

<figure>
 <img src="../../RubricPhotos/1.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>
