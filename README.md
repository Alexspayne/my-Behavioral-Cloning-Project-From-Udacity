#**Behavioral Cloning Project** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2017_02_05_19_03_01_730.jpg "Center Driving Image"
[image2]: ./examples/center_2017_02_12_16_56_56_316.jpg "Recovery Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
** I only made one edit to this file - increasing the throttle.
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 79-90) 
The model includes RELU layers to introduce nonlinearity (as a parameter), and the data is normalized in the model using a Keras lambda layer (code line 74). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 98 and 100). 
I included MaxPooling2D layers after the first three convolutional layers.

The model was trained and validated on several data sets to ensure that the model was not overfitting (code lines 13-53). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 107).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, avoiding alternate paths, and a lap around a different track.

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to [NVidia's model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I thought this model might be appropriate because NVidia got good results and thier model had the same inputs(dashcam image and steering angle) and outputs (steering angle).

From the outset, my first problem was lack of memory. NVidia's model required too much memory straight out of the box.
In order to handle that, I made a few modifications.
* Cropping the input image during preprocessing.  This was recommended in the lesson.
* Lowering the batch size. Originally I had a very large batch size, but found that I could fit a batch size of 300.


To combat the overfitting, I modified the model so that it included two dropout layers and three max pooling layers.
I also added data from the second track to my training set.
With that, my error on the validation set was actually less than the training set.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.  It ran into the lake, and it tried to take the dirt path. 
To improve the driving behavior in these cases, I recorded recovery data to specifically handle those parts of the track.

At the end of the process, the vehicle is able to drive autonomously around the track at full throttle without leaving the road.
It was also to drive around the track in both directions.

####2. Final Model Architecture

The final model architecture (model.py lines 71-104) consisted of a convolution neural network with the following layers and layer sizes:
* 5x5 convolutional layer of depth 24 with relu activation.
* Max Pooling 2D layer with a 2 by 2 stride.
* 5x5 convolutional layer of depth 36 with relu activation.
* Max Pooling 2D layer with a 2 by 2 stride.
* 5x5 convolutional layer of depth 48 with relu activation.
* Max Pooling 2D layer with a 2 by 2 stride.
* Two 3x3 convolutional layers of depth 64, each with a relu activation.
* A flattening layer
* 100 node fully connected layer with relu activation
* Dropout layer with 30% dropout
* 50 node fully connected layer with relu activation
* Dropout layer with 30% dropout
* 10 node fully connected layer with relu activation
* 1 node output layer (Steering angle)


####3. Creation of the Training Set & Training Process

##### Control

I used the beta version of the simulator to collect the training data.
The reason that I chose to use the beta simulator is that it allowed for mouse control to steer the car.
I made a control scheme for my steam controller to map it's gyroscope to horizontal mouse movement.
After experiementing with the sensitivity and the dampening, I was able to easily steer the car smoothly across the track.

I feel like this gave me much better quality data because it allowed me to easily stay close to the center of the track.
The alternative control scheme would be to snap between -25, 0, and 25 degrees for steering.
Using the controller, I was able to keep a relatively constaint steering angle around each different turn.

##### Recording

To capture good driving behavior, I first recorded a few laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the model would learn how to return to the center of the road when it got too close to the edge. These images show what a recovery looks like:

![alt text][image2]

I recorded a single lap on track two in order to help generalize.
I also used the center images from the training data provided by Udacity.

To augment the data sat, I also flipped images and angles to help generalize the model.
This way I only had to train recovery data from one side of the road. Flipping the images allowed right side recovery data to be used to learn how to recover from the left side of the track.

After the collection process, I had 25786 data points.
I preprocessed the data by cropping it. The cropping was actually built into the model though.
I didn't need to do any other preprocessing on the data.

I finally randomly shuffled the data set and put 20% of the data into a validation set.
Both steps were handled for me using the Keras framework by passing parameters to the fit() function. (model.py lines 116 and 117)

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4.  2 epochs was a bit too few, and 10 took too long.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

