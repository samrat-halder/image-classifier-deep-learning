# bottle-image-classifier
Image Classification with LeNet and AlexNet

## Overview
A 5-class liquid amount classification task.

## Dataset
### Background information
The bottle dataset was collected during ECBM E4040 Fall 2016. There are 2 types of bottles: coke bottle and water bottle, and the students were asked to take pictures with their cellphones. A post-processing was done to make sure each picture has the same size.

There are 5 classes in total:

0% (labeled as 0)
25% (labeled as 1)
50% (labeled as 2)
75% (labeled as 3)
100% (labeled as 4)
All those labels are visual estimation of the actual amount.

### Data organization
Training set: 15000 images in total, all the classes are balanced(3000 images per class). Data available on kaggle.
Test set 3500 images in total
## Models
I implemented a modified LeNet and AlexNet with 85% and 91% accuracy on test data respectively
