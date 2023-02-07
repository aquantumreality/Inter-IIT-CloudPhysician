# Project Title

A brief description of your project, including the purpose and goals of the project.

## Table of Contents

- [Brief](#Brief)
- [Models](#Models)
- [Training Epochs](#Training Epochs)
- [Hyperparameter-tuning](#Hyperparameter-tuning)
- [Pipelines](#Pipelines)
- [Authors](#Authors)


## Brief of our Work
As a part of our approach to the PS, we started with a YOLOv8 to segment the monitor screen from the rest of the background. But we had identified one problem here - drawing non-rectangular bounding boxes, which YOLO is inherently incapable of doing. This inspired us to build our own model for the same, inspired by the loss function used in YOLO - GayNet (Gains Above YOLO).

To solve the problem of predicting non-rectangular bounding boxes, we propose the GayNet model, in this we first used a mobilenetv2 backbone that will help us in extracting features from the images, we particularly choose mobilenetv2 because the evaluation was on the CPU inference time which motivated us to be as quick as possible, we took the output feature maps of the mobilenetv2 model, to improve the accuracy we extract five feature maps with different spatial resolutions from the backbone and do global average pooling and resize them to the same size; we then stack 3 fully connected layers on top of this extracted feature map, the first fully connected layers gives us the corner points of the quadilateral bounding box, the second fully connected layer will give us N - 4 points, that are equally distributed and equispaced among the four sides(Where N is the total number of points we are predicting).




LDRNet - write up 






## Models



## Training Epochs



## Hyperparameter-tuning


## Pipelines




## Authors
