# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[bar-chart-training-set]: ./examples/bar-chart-training-set.png "Class Counts - Training Set"
[augmented-image]: ./examples/augmented-image.png "Augmented image"
[image1]: ./test-images/image1.jpg "Speed limit (70 km/h)"
[image2]: ./test-images/image2.jpg "Keep left"
[image3]: ./test-images/image3.jpg "Right-of-way at the next intersection"
[image4]: ./test-images/image4.jpg "Yield"
[image5]: ./test-images/image5.jpg "No entry"
[image6]: ./test-images/image6.jpg "Bicycles crossing"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) and the exported [html document](https://github.com/nlandjev/udacity-carnd-p2/blob/master/Traffic_Sign_Classifier.html) (which you would have to download as GitHub can't display it inline)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy to calculate the sizes of the datasets and the shapes of the items in them by selecting the appropriate dimensions returned by the `shape` method of the respective dataset.

* The size of training set is 34799
* The size of the validation set is 4410 (calculated in the notebook but not printed out)
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

After loading the class names with pandas, I created a 3x3 plot with random images from the training set and their corresponding classes as titles to get an overall feel for the data. I also created two bar charts with the number of samplpes per class for the training and validation sets respectively. This is what the chart for the training set looks like:

![Class Counts - Training Set][bar-chart-training-set]

The imbalance in the number of items per class suggests adding more images for classes with fewer images might be beneficial (more on that later).

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The only preprocessing step I found to improve the results was normalizing the data by subtracting the mean and dividing by the standard deviation which were calculated with numpy from the images in the training set.

I considered turning the images to grayscale but I decided against it as I feared too much information would be lost. 

I also generated augmented images for classes with few samples in the training set but did not include them in the final training as I found they hurt the accuracy and did not help much with overfitting. It is however possible that with more experimentation one can find hyperparameters that will lead to better accuracy. Here is the approach I took with regards to data augmentation (the code is included in the notebook):

1) For each class with less than 500 samples - select random images from that class and add augmented versions of them to the training set so that each class has at least 500 samples.
2) Each augmentation consists three steps, which I implemented with opencv:
	* rotating the image by a random number of degrees in the range [-15, 15]
	* shift the image horizontally and vertically by a random number of pixels in the range [-6, 6]
	* warp the image by performing a perspective transform

Here is an example of an augmented image:
![Augmented image][augmented-image]

The notebook contains plots of a random image showing different augmented versions, obtained by applying one or more of the transofrmations.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is the LeNet architecture with a couple of slight modifications:
1) Changed number of channels for the input image as I am using all 3 channels
2) Changed number of outputs to 43 as we are classifying 43 classes.
3) Added dropout after each layer to reduce overfitting.

The following table provides a summary of all the layers.

| Layer         		|     Description	        								| 
|:---------------------:|:---------------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							            | 
| Convolution 5x5     	| 6 kernels, 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU                  |                                                           |
| Dropout				| keep_prob=0.8									            |
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6	            |
| Convolution 5x5     	| 16 kernels, 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU                  |                                                           |
| Dropout				| keep_prob=0.8									            |
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16		            |
| Flatten				| outputs 400x1												| 
| Dense					| inputs 400, output 120									|
| RELU                  |                                                           |
| Dropout				| keep_prob=0.8									            | 
| Dense					| inputs 120, output 84										|
| RELU                  |                                                           |
| Dropout				| keep_prob=0.8									            | 
| Dense					| inputs 84, output 43										|
| RELU                  |                                                           |
| Dropout				| keep_prob=0.8									            | 
| Softmax               |															| 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I started out with the code from the LeNet lab and made a few modifications.
1) I added L2 regularization by adding the L2 loss of all the weights in the network. The best results were achieved with a beta parameter of 0.001.

2) I substituted SGD with the Adam optimizer as it consistently achieved higher accuracy.

3) I used cosine annealing to reduce the learning rate with each batch. This achieved better results than just using a constant learning rate. I also tried cosine annealing with restarts but found out that resetting the learning rate would just undo several epochs of optimization and decided against using it in the final training.

After experimenting with various hyperparameters, I achieved the best results with the following values:
* initial learning rate: 0.01
* epochs: 40
* batch_size: 128
* dropout: 0.8 (activations preserved)
* beta: 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.975
* test set accuracy of 0.932

I started out with LeNet and found out that the initial results were good enough to iterate from there. It is also worth noting that the original task for LeNet - digit recognition - is similar enough to traffic sign classification, that I assumed it would be a good fit. There was slight overfitting on the training set which I decided to counter by adding regularization and dropout. I also decided against using a more complex architecture as it would probably make the overfitting problem worse and would be difficult to implement correctly.

I experimented with increasing the dimensions of the network e.g. adding more convolutional kernels and increasing the size of the dense layer but these changes did not improve the accuracy, so I decided to stick with LeNet.

After adding regularization and dropout, I experimented with different learning rates and found out that using cosine annealing with a starting learning rate of 1e-2 performed best. 

At this point I implemented the data augmentation procedure described above and found out that while this helped with the overfitting a bit (the model overfit later in the training), it could not achieve the same accuracy and was also a lot slower due to the higher number of training examples and because it forced the starting learning rate to be smaller. 

I then spent more time tweaking the existing hyperparmeters - starting learning rate, number of epochs, dropout and beta (for l2 regularization) and arrived at a validation accuracy of 0.975. What was somewhat surprising was the fact that changing the dropout parameter did not result in a less overfitting but in changes in the accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web (cropped and resized, I deleted the non-resized versions, sorry :/ ):

![Speed limit (70 km/h)][image1]
![Keep left][image2]
![Right-of-way at the next intersection][image3]
![Yield][image4]
![No entry][image5]
![Bicycles crossing][image6]

To be fair, as the signs are clearly visible in all of the images and the contrast is consistently very good, the model should not have a problem with any of them. Nevertheless:

* The first image might be difficult to classify because the red is slightly brighter than in most of the training set images I looked at and because the sign does not fit the whole frame;

* The second image might be difficult to classify because it looks similar to the 'Keep right' sign and the small image size might not be sufficient to differentiate between the two;

* The third and sixth images might be difficult to classify because they are part of a group of signs that are differentiated only by the image inside. Otherwise they have the same shape and colors (white triangular sign with red border). Again, the small image size might be a problem;

* The fourth image might be difficult to classify because the sign doesn't fill the whole frame and it is shot from below, distorting the image shape;

* The fifth image might be difficult to classify because it is shot from the side, distorting the image shape;

* The sixth image might also be difficult to classify because there is a metal border around it;

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        				|     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| Speed limit (70 km/h)					| Speed limit (30 km/h)							| 
| Keep left    							| Keep left										|
| Right-of-way at the next intersection | Right-of-way at the next intersection			|
| Yield									| Yield											|
| No entry					     		| No entry						 				|
| Bumpy Road							| Bicycles crossing    							|


The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66.7%. This is less than the accuracy on the test set, which was 93.2%. However, it should be noted that in both cases where the prediction was wrong, the predicted sign is visually similar to the correct sign and the differences are only in the symbol inside the sign - 70 vs 30 km/h, bumps vs bicycle). 

Looking at the precision and recall of the affected classes does not suggest any specific problems related to them - the biggest problem is the 50-60% precision of the following signs
	* Speed limit (20 km/h)
	* Go straight or left
	* Pedestrians
	* Roundabout mandatory

as well as the ~40-60% recall for the following signs:
	* Road narrows on the right
	* Pedestrians
	* Beware of ice/snow

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the four correctly predicted images, the model predicted the correct class with probability 1 (up to 3 decimal places).

For the first image, incorrectly classified as Speed limit (30 km/h), the correct class has probability 4.6% with the top five probabilities as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .801         			| Speed limit (30 km/h)	  						| 
| .067     				| Speed limit (20 km/h)							|
| .046					| Speed limit (30 km/h) <- correct class		|
| .027	      			| Traffic signals				 				|
| .022				    | General caution     							|


For the last image, incorrectly classified as Bicycles crossing, the correct class has probability 7.8% with the top five probabilities as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .902        			| Bicycles crossing		  						| 
| .078    				| Bumpy road  <- correct class					|
| .015					| Slippery road									|
| .004	      			| Children crossing				 				|
| .000				    | Speed limit (60km/h)   						|


