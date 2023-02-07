# Team 13 - Cloud Physician, Inter IIT Tech Meet 11.0 (Team ID - 13, Secondary ID - 26)

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
- [Pipelines](#Pipelines)
- [Models](#Models)
- [Leveraging-data](#leveraging-data)
- [Inference Time](#inference-time)
- [Hyperparameter-tuning](#Hyperparameter-tuning)
- [Possible Future Work](#possible-future-work)


## Brief of our Work

The Vital Extractor model built by Team-13 leverages segmentation, object detection, colour segmentation and edge detection to achieve state of the art results for detecting various kinds of vitals namely Heart Rate, Blood Pressure, SPO<sub>2</sub> and Respiration Rate with their corresponding graphs (if any) from the monitor.

Unlike existing object detectors we make use of a novel detector that gives non-rectangular bounding boxes as well. Using this as the base of our object detector, we use state of the art methods (such as YOLO) to do color-based detection and refinement of the vitals present on screen. To top it all off, we make use of a custom edge detector to digitise the graphs present on the monitor screen yielding promising results.

As a part of our approach to the PS, we started with a YOLO to segment the monitor screen from the rest of the background. But we had identified one problem here - drawing non-rectangular bounding boxes, which YOLO is inherently incapable of doing. This inspired us to build our own model for the same, called **GAYnet** (Gains Above YOLO) - inspired by the loss function used in YOLO. 

We tried out several existing OCR frameworks for extracting text from the segmented monitor image. Since this was not able to yield desired results, we used YOLOv5 for localization of vitals on the screen and then leveraging **paddleOCR** post-vital localization inherently accelerated our pipeline and reduced the inference time from around 20 seconds to less than 2 seconds. 

We tried **SRResnet** and several other resolution techniques for improving the resolution of the segmented monitor image, and then adaptive thresholding as it was yielding better results. But since there was loss of information, this was not a part of out final pipeline. Following this, we realized that it wasn't really required in our pipeline as well. 

For digitizing the heart rate, we also tried out Vision Transformer models like **DINO** (for crack detection) but using our custom edge detector gave us more promising results in comparison and more robustness, reduce time complexity.  


## Pipelines
Here is our proposed pipeline for the same:

![Pipeline](https://github.com/aquantumreality/Inter-IIT-CloudPhysician/blob/main/pipeline.png "Pipeline")

## Models

### Monitor Segmentation

To solve the problem of predicting quadilateral bounding boxes, we propose the Gains-Above-YOLO-Net or as we like to call it the <b>GAYnet</b> , in it we first used a <b>MobileNetv2</b> backbone that will help us in extracting features from the images, we particularly choose **MobileNetv2** because the evaluation was on the CPU inference time which motivated us to be as fast as possible, we took the output feature maps of the **MobileNetv2** model. To improve the accuracy we extract **five** feature maps with different spatial resolutions from the backbone and perform a Global Average Pooling and resize them to the same size. We then stack three fully connected layers on top of this extracted feature map, the first fully connected layer will give us the corner points of the quadilateral bounding box, the second fully connected layer will give us N - 4 points, that are equally distributed and equispaced among the four sides (where N is the total number of points we are predicting).

For loss functions we first used the Mean Squared Error but quickly realized precision localization and segmentation tasks are fundamentally difficult for standard Deep Convolutional Neural Network designs to complete. This happens because the final convolutional layer only includes the most prominent aspects of the entire image. These features lack the data necessary for pixel-level segmentation, despite being very helpful for classification and bounding box detection so towards the end we predict the corners in a line-prediction fashion.We identify the equal-division points on the lines in addition to the four corner points, allowing the labels to be created automatically with no further human input needed.

We call this loss function **Monitor Loss** which can be broken into two parts, one is that check the **parallelism** of the edges of the monitor and one that maintains the **equidistance** between the corners of the monitor.

$$\mathcal{L} _{monitor} = \varphi \mathcal{L} _{parellel} + \eta \mathcal{L}_{eq}$$

Using this novel loss function significantly improved the performance of our model but we were still seeing that the model is not exactly predicting the correct boundary points even though the loss has almost converged. This can be explained by seeing that we were working with normalized coordinates so suppose if the $L_{1}$ error is of the order of $10^{-2}$ then the MSE will make it $10^{-4}$ which will not leave enough gradient to flow back. This  will thus lead to a sluggish training and and $L_{1}$ error of about $0.01$ which is not good at all because finally we will be scaling it by 400 to get the exact point coordinates, thereby increasing the actual error by a significant amount.

So towards this we again proposed a **novel loss function** :
          $$\frac{\lambda log(1+(L_{1})^{2})}{N}$$
The Idea behind this was first we will train our model with only the MSE loss until it converged and then we will switch to the Log Loss, for which now the error will be very small as compared to 1 so we could effectively write log(1 + x) as x and here we could set lambda=1000 for making the model focus more on the 3rd decimal place, we also squared the final error because we only wanted positive error. This led to significant improvements in our performance, and the points that we were now predicting were almost perfect.

- #### IOU Loss

We also added IOU loss function which was $\frac{|A_{pred} - A_{actual}|}{|A_{pred} + A_{actual}|}$ to directly improve on the IOU metric. Apart from this we saw that our model was not generalizing properly and  also had not used the unlabelled dataset as such, hence training our models on the outputs of yolov8 from the unlabelled dataset significantly improved our performance. Since our bounding boxes are quadrilateral, a better IOU loss was achieved when compared to yolov8 which is only capable of handling rectangular bounding boxes.

- #### Classification Loss
We explored a classification loss to better the accuracy, since MSE loss function is inherently a regression loss function. To make the mobilenet learn useful feature representation, we introduced Binary cross Entropy classification loss with some random images that did not contain screen as the negative train samples. 

- #### Planar Homography

An inherent problem with the datasets was the oblliqueness of the monitor screens which led to issues in optical character recognition of the vitals.Therefore, we used classic Computer vision techniques for warping the oblique images onto a plane to make it easier for the OCR to recognize the vitals accuractely.

### Vital Extraction (YOLOv5)

Post usage of our novel monitor segmentation model, yolov5 was leveraged to detect bounding boxes around the vitals. We took around 200 images from the unlabelled and 300 images from the classification dataset, manually annotated them using Roboflow and augmented those 500 images to get a dataset of 1200 images and trained YOLOv5 on this dataset. Owing to different color characteristics in different monitor types, we used other features such as the presence of the indicators like **HR**, **RR**, **SPO<sub>2</sub>**, etc. that were common across all monitors to improve the robustness and accuracy of our model.

### OCR related work (PaddleOCR)

In this stage, we used the **PaddleOCR** library for text extraction from the bounding boxes we had. In parellel we also extracted the dominant colour in each bounding box. 

(About PaddleOCR: PaddleOCR is an optical character recognition (OCR) toolkit developed by PaddlePaddle (PArallel Distributed Deep LEarning), an open-source deep learning platform. It provides a comprehensive set of OCR solutions, including text recognition, table recognition, form recognition, and license plate recognition. The toolkit is built on PaddlePaddle, a flexible, easy-to-use, and high-performance deep learning framework, making it possible to train custom models to meet specific OCR requirements.)

We then used an algorithmic approach based on range of text values taken by the vital signs and the possible colors that a vital could take to refine our results from above. The dominant color from each YOLO detected frame was extracted from the hsv space of the frame. Firstly, HSV limits for each color
were defined and a mask was created for the same. This mask was used to quantify the area of the frame containing that particular color. 
Based on the amount of a color present, the dominant color is chosen for each frame.

### Digitizing the Graph

We used concepts from the Canny Edge detection algorithm (like non-maximum suppression) to extract out the x and y coordinates of the graph and plot it using matplotlib. We used scipy.interpolate to interpolate the extracted points (spline interpolation) and generate the digitized graph. 

## Leveraging data 
- The 2000 images from the monitor segmentation dataset were used to train our GAYnet segmentor. 
- Monitor images from the classification dataset were used to train the yolov5 vital extractor.
- The Unlabelled data was used to a large extent. Firstly, a YOLO model was used to generate annotations for 750 images from the unlabelled dataset that had unseen features. These images were added to the 2000 monitor segmentation images for training which greatly improved the performance of our Gaynet model
- 200 monitor screens covering all varieties from the unlabelled dataset were again annotated for each specific vital 
and added to the training data for our YOLOv5 vital extractor. This made our vital extractor more robust and versatile.

## Inference Time

On an average, the CPU inference time is lower than 2 seconds and consistently around 1.5 seconds. 
```
import time
from timeit import default_timer as timer

path = '/content/gdrive/MyDrive/Vital_extraction/seg_test/hcgvijayawada_icu_mon--8_2023_1_5_1_27_36.jpeg'

start_time = timer()

results = Pipeline(path, model, model_yolo, reader)

end_time = timer()

print(f'Inference time: {(end_time-start_time):.2f}')
```

```
Inference time: 1.57
```
![impl_ss](https://github.com/aquantumreality/Inter-IIT-CloudPhysician/blob/main/inferfinalv2.ipynb%20-%20Colaboratory%20-%20Google%20Chrome%2007_02_2023%2011_02_18%20PM.png)

## Hyperparameters:


For GAYnet
```
backbone_alpha: 1.4
points_size: 8
batch_size: 16
epochs: 400
img_folder_path: 
- "/kaggle/input/vitalextr2/images"
annotations_path:
- "/kaggle/input/vitalextr2/labels.csv"
class_list:
- 1
rate: 0.001
gamma: 0.5
bounds:
- 50
- 100
- 150
- 200

loss:
loss_ratio: 50
log: 100
giou: 50
slop_loss_ratio: 0.1
diff_loss_ratio: 0.1
class_loss_ratio: 1
```

For YOLOv5

```
lr0=0.01 , lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8,
warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0,
fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, 
shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
```

## Possible Future Work

One of our ideas was to use a teacher-student mechanism between YOLOV8 and GAYnet for training GAYnet. 

