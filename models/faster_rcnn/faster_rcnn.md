# Faster-RCNN

This documents just for myself understanding for faster-rcnn. Hope it can help you understand this network well also.

[_Faster-rcnn_](https://arxiv.org/pdf/1506.01497.pdf) is a so popular network to be adopted on the commercial and industrial area wildly.

[This](https://zhuanlan.zhihu.com/p/31426458) explain is so good to beginner to understand Faster-RCNN well.

![](https://cdn-images-1.medium.com/max/1600/1*e6dx5qzUKWwasIVGSuCyDA.png)

![](https://pic4.zhimg.com/80/v2-e64a99b38f411c337f538eb5f093bdf3_hd.jpg)

It includes 3 parts:

- Basic feature extractor network (E.g Vgg-16, in this paper): To extract features for generating region proposals

![](https://adeshpande3.github.io/assets/zfnet.png)

<center>ZF net</certer>

![](https://qph.fs.quoracdn.net/main-qimg-83c7dee9e8b039c3ca27c8dd91cacbb4) 

<center>VGG 16</center>


- Region Proposal Network (RPN): On the top of the _basic feature extractor_, can simultaneously regress region bounds and objectness scores.

![](https://pic3.zhimg.com/80/v2-1908feeaba591d28bee3c4a754cca282_hd.jpg)

- Fast RCNN: To do regression of **Coordinate of Bounding Box** and **Class Score** for each **Proposal Region**




## Architecture

#### 1. Basic Feature extract:  


* [x] [vgg-16/19](../frontend/vgg.py)


#### 2. RPN
* [x] [GET RPN](../../data_utils/data_generator/FasterRcnn/Rpn_utils#76)  
* [ ] RPN to ROI

#### 3. Fast RCNN
**TO DO**
