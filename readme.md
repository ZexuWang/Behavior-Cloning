
# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_architecture.png "Model architecture"
[image2]: ./examples/left_dive.png "left driving"
[image3]: ./examples/right_drive.png "right driving"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* readme.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works and how the preprocessing of training data are implemented.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The architecture of the model can be found in the below summary

![alt text][image1]



My model consists of a lambda layer that normalize the image data(model.py line 119), convolution neural network layers with 5x5 filter sizes with depths between 32, 64 and 128 (model.py lines 119-122), convulutional neural network layers with 3x3 filter sizes with depth between 64 and 32(model.py lines 123-125), a flatten layer(model.py line 126), a dropout layer with 0.5 dropout rate to reduce overfitting(model.py line 127), densely connected layers with sizes 100,50,10,1(model.py line 129-132)

The above convolutional lays and dropout layer are connected to RELU layers to introduce nonlinearity. 

A generator was implemented to reduce memory usage(model.py line 104-110)


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 127). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 135-139). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 134).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of clockwise driving and counter clockwise driving to keep the balance of steering angles of the vehicle.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the base model from the traffic sign classifier from previous porject and make modification on top of the base model.

My first step was to use a convolution neural network model similar to the traffic sign classifier. I thought this model might be appropriate because the traffic sign classifier take the images as input and gives the identified traffic sign number as output which is quites similar to the behavior cloning networks which takes also the images as input and gives steer angles as output.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Since the images from the vehicle simulator is much larger than the one in the traffic sign classifier, I added more convolution layers and then I see my model has a decrease in traning loss while the validation loss is not decreasing as expected, which means there's overfitting problem during the training.

To combat the overfitting, I modified the model by adding dropout layer with 0.5 dropping out rate.Then I added flatten layer and several densely connected layers such that the output has the size we want.

The final step was to run the simulator to see how well the car was driving around track one. There were one spot where the vehicle run into the ledge, to improve the driving behavior in the case, I slightly change the number of training epochs.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded two laps on track one where the first lane is diven clockwise and the second lane is driven counter clockwise. To capture more features of how the vehicle behaves when running to the edges, both driving trails are implemented by mainly driving in the center of the lane with some modification driving by first run to the direction of the road edge and then drive back. Here is an example image of left driving:

![alt text][image2]



To augment the data sat, I also have the second lane of driving but in difference direction and here is an example of right driving:

![alt text][image3]



After the collection process, I had 4744 number of data points. I then augumented this data by copying the ones which non-zero angle steerings. Details of data augumentation can be found in behavior_cloning_script.ipynb.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 by trail and error with criteria that the epochs should be keep as small as possible with decreading training and validation losses. I used an adam optimizer so that manually training the learning rate wasn't necessary. The final result of the trained 'driver' seems to be okay with handling different situations of steering.

