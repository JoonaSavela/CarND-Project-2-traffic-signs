# **Traffic Sign Recognition** 

## Writeup Template

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./car_p2_bar_chart.png "Visualization"
[image2]: ./car_p2_grayscale.png "Grayscaling"
[image3]: ./car_p2_clahe_norm.png "Clahe and normalization"
[image4]: ./my_german_traffic_signs/german-traffic-signs_1.png "Traffic Sign 1"
[image5]: ./my_german_traffic_signs/german-traffic-signs_2.png "Traffic Sign 2"
[image6]: ./my_german_traffic_signs/german-traffic-signs_3.png "Traffic Sign 3"
[image7]: ./my_german_traffic_signs/german-traffic-signs_4.png "Traffic Sign 4"
[image8]: ./my_german_traffic_signs/german-traffic-signs_5.png "Traffic Sign 5"
[image9]: ./car_p2_preprocess.png "Preprocessing"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Putteri11/CarND-Project-2-traffic-signs/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python standard library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a normed bar chart showing how the data is divided between the classes.

![alt text][image1]

As can be seen, the data is very unevenly divided, which will probably lead to some overfitting of the model.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it is easier for the model to learn from grayscale images than color images.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

I also decided to apply Contrast Limited Adaptive Histogram Equalization (or CLAHE for short) to the grayscaled images in order to make important properties stand out and also to make the images more similar with each other.

As a last step, I normalized the image data because the model learns normalized data better than unnormalized data.

Here is an example of the same grayscaled traffic sign image, first with CLAHE applied (left), and then normalized (right).

![alt text][image3]

I did not genrate any new data nor did I augment existing images.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Preprocessed image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, output 28x28x6 	|
| RELU					|												|
| Average pooling	      	| 2x2 stride and kernel, valid padding, output 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, output 10x10x16			|
| RELU					|												|
| Average pooling	      	| 2x2 stride and kernel, valid padding, output 5x5x6 				|
| Flatten    |   output 400  |
| Fully connected		| output 100        									|
| RELU					|									    			|
| Fully connected		| output 70        									|
| RELU					|									    			|
| Dropout  |   keep probability (training) 0.6   |
| Fully connected	(output/logits)		| output 43       									|
 
I used the LeNet architecture as a basis for my model, but made a few changes. I changed the Max pooling into average pooling, because there is some loss of information in Max pooling, and they both do essentially the same thing. I also altered some the dimensions of the original model, and added dropout in order to prevent overfitting.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer, a batch size of 80, and 15 epochs. The learning rate was 0.001. I found that a smaller batch size and a larger number of epochs yielded in better results, otherwise the training is pretty much identical to the training in the LeNet architecture.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.962
* test set accuracy of 0.936

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I chose the LeNet architecture as the basis of my model, because it was said to be a good starting point for this project.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first and the second images might be difficult to classify because of the peculiar lighting conditions on the images. The third image might be difficult because it's quite tilted to the left. The fourth is probably the easiest to classify, thanks to preprocessing of the images, but is still somewhat unclear and too much lit. The fifth and final image might be difficult to classify because it's partly covered by an obstacle, and because it's poorly represented in the training data, as seen from the bar chart above (pedestrians, label 27).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30 km/h)      		| Speed limit (30 km/h)   									| 
| Dangerous curve to the left     			| Dangerous curve to the left 										|
| Road Work					| Road Work											|
| Traffic signals	      		| Traffic signals					 				|
| Pedestrians			| Pedestrians      							|


The model was able to correctly guess all of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


