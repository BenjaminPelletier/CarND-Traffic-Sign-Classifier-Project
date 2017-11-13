#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./export/exploration1.png "Examples of each class"
[image2]: ./export/exploration2.png "Class distribution in dataset"
[image3]: ./export/exploration3.png "Similarity of training images"
[image4]: ./export/yuv.png "Y channel only"
[image5]: ./export/signs.png "Five traffic signs"
[image6]: ./export/predictions.png "Prediction results"
[image7]: ./export/top5predictions.png "Top 5 predictions"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!  Here is a link to my [project code](https://github.com/BenjaminPelletier/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) and the [rendered output](https://github.com/BenjaminPelletier/CarND-Traffic-Sign-Classifier-Project/blob/master/export/Traffic_Sign_Classifier.html).

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 32x32 RGB images
* The size of the validation set is 4410 32x32 RGB images
* The size of test set is 12630 32x32 RGB images
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It shows representative images from each class:

![Examples of each class][image1]

I noticed that most classes appear next to each other in the dataset, but they are not ordered according to class index:

![Class distribution in dataset][image2]

At least some training images seem nearly identical with slight cropping differences:

![Similarity of training images][image3]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the image data to zero-mean between -1 and 1 to limit the amount of adjustment that would likely be necessary during training for the weights (e.g., the biases wouldn't have to compensate for the mean of the dataset).

This is the only preprocessing I performed initially.  However, my model initially did not train well: it asmpytoted to only high 70's percent validation accuracy.  The training accuracy was always near 1 which indicated overfitting.  Even after adding some dropout (although perhaps not fully correctly) and L2 normalization, I didn't achieve the desired 93%.  I referred to "Traffic Sign Recognition with Multi-Scale Convolutional Networks" and read that the authors found that training only on the Y component of YUV-decomposed images actually produced better results, so I replicated that result by proprocessing the images to select only the Y component for each.  Here is what the images look like when their U and V channels have been removed:

![Y channel only][image4]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer | Description | Parameter count |
|:---------------------:|:---------------------------------------------:|:---:|
| Input         		| 32x32x1 Grayscale image   							| |
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x6 	| 156 |
| RELU					|												|
| Max pooling 2x2	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									| 2416 |
| RELU					|												| |
| Max pooling 2x2	      	| 2x2 stride,  outputs 5x5x16 				| |
| Flatten | Outputs 400-length vector | |
| Fully connected		| Outputs 120-length vector       									| 48120 |
| RELU					|												| |
| Dropout | Probability 50% during training | |
| Fully connected		| Outputs 84-length vector       									| 10164 |
| RELU					|												| |
| Dropout | Probability 50% during training | |
| Fully connected		| Outputs 43-length vector       									| 3655 |
| Softmax				| Produces probabilities for each of the 43 classes        									|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained my model using an Adam optimizer with learning rate 0.001 over 50 epochs with a batch size of 64.  I tried modifying these values somewhat between hyper iterations, but these values qualitatively seemed to produce the fastest and most reliable convergence.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 95.6%
* test set accuracy of 94.4%

I chose to use the LeCun architecture for classifying MNIST handwriting samples as a base architecture.  I believed it would be relevant because it also is intended to classify small images based on key features in those images (and also because the Udacity lessons suggest it as a starting point).

I originally modified this architecture to accept 3-channel RGB images because I figured hue would be an important clue in classifying the images.  I considered preprocessing to HSV, but discarded that idea because the cyclic hue would be difficult to use properly at its modulus point.  Unfortunately, this modified architecture produced only high 70's percent validation accuracy despite training accuracy of nearly 100%.  I diagnosed the problem as overfitting and therefore added dropout to the fully-connected layers.  This did not improve the results substantially, so I tried also adding dropout to the convolutional layers.  This did not improve the results either, so I added L2 regularization to the loss function.  This failed to improve the results substantially as well.

At this point, I referred to the LeCun paper linked in the lesson and found that they decomposed the image into YUV, so I tried that preprocessing step instead of the dropout and regularization.  Again, this failed to produce substantially better results.  Finally, I selected only the Y channel to preprocess the input images to grayscale.  This improved results noticeably, but not yet sufficiently.  I re-added dropout on the fully-connected layers and this yielded results of the desired quality. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Five traffic signs][image5]

These signs were taken as the first 5 relevant results from a Google search for 'german traffic signs'.  They were originally larger and higher resolution, but were downsampled before this image was generated.  The original code to retrieve the images directly from their sources is included in the notebook.

The first sign may be difficult to classify because the background is noisy (there is clutter that could be mistaken for features of interest).

The second sign may be difficult to classify because the shape of the 6 shares many features with both a 5 and an 8.

The third sign may be difficult to classify because the pedestrian figures are not clearly visible to a human as pedestrians, and so may be confused with shadows or other irrelevant features by the neural network.

The fourth and firth signs may be difficult to classify because they both look very similar with a predominantly black figure in the middle of a triangular sign.  The black figures are only subtly different from each other and might easily be confused but for a very small number of pixels.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![Prediction results][image6]

In this particular training session, the model was able to correctly guess 4 of the 5 traffic signs giving an accuracy of 80%.  This is substantially below the test set accuracy of around 94%.  This could simply be due to a small sample size and therefore insufficient statistical significance of the difference between 80% and 94%, but I have run the entire training process a few different times (for other reasons) and observed that the accuracy on these 5 images varies substantially.  I observed one run where all 5 images where classified correctly to very high confidence, runs where the incorrect image was the 60 kph (misclassified as 50 or 80), and a run where both the fourth and fifth images were both classified as general caution.

I believe this suggests that my model is still overfitting on the training data, and there are many possible ways in which this overfit may occur.  Going back and revising my network to improve on the test sets (these 5 images and the original test set) would be cheating however because then I would essentially be fitting to the test data.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 8th cell of the Ipython notebook (single prediction) and 10th cell (top 5 predictions).  Here are the top 5 predictions for each image, along with the probability of each prediction:

![Top 5 predictions][image7]

The model is exceedingly certain about all of its top predictions, except for the speed limit.  In that case, it was very sure of its top 2 possibilities, but was uncertain between them and ended up picking the incorrect option.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


