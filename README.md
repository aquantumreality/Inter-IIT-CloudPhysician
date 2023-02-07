# Team 13 - Cloud Physician, Inter IIT Tech Meet 11.0



## Table of Contents

- [Brief of our Work](#brief-of-our-work)
- [Models](#Models)
- [Training Epochs](#training-epochs)
- [Hyperparameter-tuning](#Hyperparameter-tuning)
- [Pipelines](#Pipelines)


## Brief of our Work

The Vital Extractor model built by Team-13 leverages segmentation, object detection, colour segmentation and edge detection to achieve state of the art results for detecting various kinds of vitals namely Heart Rate, Blood Pressure , SPO<sub>2</sub> and Respiration Rate with their corresponding graphs(if any) from the monitor.

Unlike existing object detectors we make use of a novel detector that gives non rectangular bounding boxes as well. Using this as the base of our object detector, we use state of the art methods(such as yolo) to do color based detection and refinement of the vitals present on screen. To top it all off, we make use of a custom edge detector to digitise the graphs present on the monitor screen yielding promising results.

## Models

- ### Monitor Segmentation

To solve the problem of predicting quadilateral bounding boxes, we propose the Gains-Above-YOLO-Net or as we like to call it the <b>GAYnet</b> , in it we first used a <b>MobileNetv2</b> backbone that will help us in extracting features from the images,  we particularly choose **MobileNetv2** because the evaluation was on the cpu inference time which motivated us to be as fast as possible, we took the output feature maps of the **MobileNetv2** model. To improve the accuracy we extract **five** feature maps with different spatial resolutions from the backbone and perform a Global Average Pooling and resize them to the same size. We then stack 2 or 3 (3 if we want to write about classification) fully connected layers on top of this extracted feature map, the first fully connected layer will give us the corner points of the quadilateral bounding box, the second fully connected layer will give us N - 4 points, that are equally distributed and equispaced among the four sides(Where N is the total number of points we are predicting).


For loss functions we first used the Mean Squared Error but quickly realized precision localization and segmentation tasks are fundamentally difficult for standard Deep Convolutional Neural Network designs to complete. This happens because the final convolutional layer only includes the most prominent aspects of the entire image. These features lack the data necessary for pixel-level segmentation, despite being very helpful for classification and bounding box detection so towards the end we predict the corners in a line-prediction fashion.We identify the equal-division points on the lines in addition to the four corner points, allowing the labels to be created automatically with no further human input needed.

We call this loss function **Monitor Loss** which can be broken into two parts, one is that check the **parallelism** of the edges of the monitor and one that maintains the **equidistance** between the corners of the monitor.

$$\L _{monitor} = \beta \L _{parellel} + \gamma \L_{eq}$$

Using this novel loss function significantly improved the performance of our model but we were still seeing that the model is not exactly prediction the correct boundary points but the loss has almost converged, this can be explained by seeing that we were working with normalized coordinates so suppose if we L1error of the order of 10^-2 the the MSE will make it 10^-4 which will not leave enough gradient to flow back and will thus lead to a sluggish training and and L1error of about 10**-2 was not good because finally we will be scaling it by 400 to get the exact point coordinates which will increase the actual error.

So towards this we again proposed a novel loss function with is 1/N*(lambda*log(1+L1error))**2,
The Idea behind this was first we will train our model with only the MSE loss until it converged and then we will switch to the Log Loss, for which now the error will be very small as compared to 1 so we could effectively write lambda*log(1 + x) as lambda*x and here we could set lambda 1000 for making the model focus more of the 3rd decimal place, we also squared the final error because we only wanted positive error, this significantly improved our performance, and the points that we were predicting were almost perfect,



## Training Epochs



## Hyperparameter-tuning


## Pipelines


