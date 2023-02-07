# Team 13 - Cloud Physician, Inter IIT Tech Meet 11.0

Patient monitoring is a crucial aspect of healthcare, as it allows healthcare professionals
to closely track a patient's vital signs and detect any potential issues before they become
serious. In particular, monitoring a patient's vitals, such as heart rate, blood pressure, and
oxygen levels can provide valuable information about a patient's overall health and
well-being. By closely monitoring vitals, healthcare professionals can quickly identify any
abnormal changes, and take appropriate action to address any concerns. Additionally,
with the increasing use of technology in healthcare, there are now many digital
monitoring systems available that can help to automate the process of tracking vitals,
making it more efficient and accurate. Overall, monitoring vitals is a critical aspect of
providing high-quality care to patients and is essential for ensuring the best possible
patient outcomes. According to the current guidelines, the nurse-to-patient ratio is 1:6,
however, in the real world, the situation is much worse.

The core problem statement is to extract Heart Rate, SPO<sub>2</sub>, RR, Systolic Blood Pressure,
Diabolic Blood Pressure, and MAP from the provided images.

## Table of Contents

- [Brief of our Work](#brief-of-our-work)
- [Models](#Models)
- [Training Epochs](#training-epochs)
- [Hyperparameter-tuning](#Hyperparameter-tuning)
- [Pipelines](#Pipelines)



## Brief of our Work

The Vital Extractor model built by Team-13 leverages segmentation, object detection, colour segmentation and edge detection to achieve state of the art results for detecting various kinds of vitals namely Heart Rate, Blood Pressure, SPO<sub>2</sub> and Respiration Rate with their corresponding graphs (if any) from the monitor.

Unlike existing object detectors we make use of a novel detector that gives non-rectangular bounding boxes as well. Using this as the base of our object detector, we use state of the art methods (such as YOLO) to do color-based detection and refinement of the vitals present on screen. To top it all off, we make use of a custom edge detector to digitise the graphs present on the monitor screen yielding promising results.

As a part of our approach to the PS, we started with a YOLOv8 to segment the monitor screen from the rest of the background. But we had identified one problem here - drawing non-rectangular bounding boxes, which YOLO is inherently incapable of doing. This inspired us to build our own model for the same, inspired by the loss function used in YOLO - GAYnet (Gains Above YOLO).

We used planar homography to 

## Models

- ### Monitor Segmentation

To solve the problem of predicting quadilateral bounding boxes, we propose the Gains-Above-YOLO-Net or as we like to call it the <b>GAYnet</b> , in it we first used a <b>MobileNetv2</b> backbone that will help us in extracting features from the images, we particularly choose **MobileNetv2** because the evaluation was on the CPU inference time which motivated us to be as fast as possible, we took the output feature maps of the **MobileNetv2** model. To improve the accuracy we extract **five** feature maps with different spatial resolutions from the backbone and perform a Global Average Pooling and resize them to the same size. We then stack three fully connected layers on top of this extracted feature map, the first fully connected layer will give us the corner points of the quadilateral bounding box, the second fully connected layer will give us N - 4 points, that are equally distributed and equispaced among the four sides (where N is the total number of points we are predicting).

For loss functions we first used the Mean Squared Error but quickly realized precision localization and segmentation tasks are fundamentally difficult for standard Deep Convolutional Neural Network designs to complete. This happens because the final convolutional layer only includes the most prominent aspects of the entire image. These features lack the data necessary for pixel-level segmentation, despite being very helpful for classification and bounding box detection so towards the end we predict the corners in a line-prediction fashion.We identify the equal-division points on the lines in addition to the four corner points, allowing the labels to be created automatically with no further human input needed.

We call this loss function **Monitor Loss** which can be broken into two parts, one is that check the **parallelism** of the edges of the monitor and one that maintains the **equidistance** between the corners of the monitor.

$$\mathcal{L} _{monitor} = \varphi \mathcal{L} _{parellel} + \eta \mathcal{L}_{eq}$$

Using this novel loss function significantly improved the performance of our model but we were still seeing that the model is not exactly ting the correct boundary points even though the loss has almost converged. This can be explained by seeing that we were working with normalized coordinates so suppose if the $L_{1}$ error is of the order of $10^{-2}$ then the MSE will make it $10^{-4}$ which will not leave enough gradient to flow back. This  will thus lead to a sluggish training and and $L_{1}$ error of about $0.01$ which is not good at all because finally we will be scaling it by 400 to get the exact point coordinates, thereby increasing the actual error by a significant amount.

So towards this we again proposed a **novel loss function** :
          $$\frac{\lambda log(1+(L_{1})^{2})}{N}$$
The Idea behind this was first we will train our model with only the MSE loss until it converged and then we will switch to the Log Loss, for which now the error will be very small as compared to 1 so we could effectively write log(1 + x) as x and here we could set lambda=1000 for making the model focus more on the 3rd decimal place, we also squared the final error because we only wanted positive error, this significantly improved our performance, and the points that we were predicting were almost perfect.


## Pipelines
Here is our proposed pipeline for the same:

![Pipeline](https://github.com/aquantumreality/Inter-IIT-CloudPhysician/blob/main/pipeline.png "Pipeline")


## Training Epochs



## Hyperparameter-tuning




## Authors
