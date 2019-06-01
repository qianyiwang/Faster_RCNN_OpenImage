# Faster-RCNN Practise using Google Open-Image dataset

## References:
* [Code Reference link](https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras)
* [Data Reference link](https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes)

## Faster-RCNN
* Using RPN (CNN) instead of selective search algorithm to propose region 
* Object detection is using CNN (VGG-16)
* Both region proposal generation and objection detection tasks are all done by the same conv networks. After getting ROI, they will be put on top of feature map. With such design, object detection is much faster.

