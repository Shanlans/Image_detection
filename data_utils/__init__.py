

## Data Augmentation

## Pascal Obj Parser

from .data_parser.pascal_voc_parser import get_pascal_detection_data


## Dataset generator

# Cifa 10 100 dataset
from .data_generator.CifarGen import CIFARDataGen

# OCR dataset
from .data_generator.OcrGen import OCRDataGen

# Faster Rcnn
from .data_generator import FasterRcnnDataGen

# Tf-record: To do, the best practice is tf-data pipeline
