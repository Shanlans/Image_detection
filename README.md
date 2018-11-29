# Image Detection

## Project Architecture

+---config  
+---data_utils  
|   +---data_generator  
|   |   +---data_augmentor  
|   |   +---FasterRcnn  
|   +---data_parser  
|   +---tf_record  
+---logs  
+---metadata  
+---models  
|   +---classification  
|   +---faster_rcnn  
|   +---frontend  
+---monitor  
+---output  
+---train_utils  
|   +---callback  
|   +---loss  
+---main.py


## Parameter

* [Common parameter](./config/common_cfg.py)
* [Faster Rcnn parameter](./config/faster_rcnn_cfg.py)


## Documentation

* [Faster Rcnn](./models/faster_rcnn/faster_rcnn.md)




