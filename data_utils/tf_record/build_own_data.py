# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts PASCAL VOC 2012 data to TFRecord file format with Example protos.

PASCAL VOC 2012 dataset is expected to have the following directory structure:

  + pascal_voc_seg
    - build_data.py
    - build_voc2012_data.py (current working directory).
    + VOCdevkit
      + VOC2012
        + JPEGImages
        + SegmentationClass
        + ImageSets
          + Segmentation
    + tfrecord

Image folder:
  ./VOCdevkit/VOC2012/JPEGImages

Semantic segmentation annotations:
  ./VOCdevkit/VOC2012/SegmentationClass

list folder:
  ./VOCdevkit/VOC2012/ImageSets/Segmentation

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import math
import os
import os.path
import sys
# import build_data
import tensorflow as tf

from dataset.data_create import build_data
FLAGS = tf.app.flags.FLAGS

_NUM_SHARDS = 3



def _convert_dataset(filepath, outputdir, train_data=True, image_format='png', label_format='png',id_to_className=None,job_type='Classification'):
    """Converts the specified dataset split to TFRecord format.

    Args:
      dataset_split: The dataset split (e.g., train, test).

    Raises:
      RuntimeError: If loaded image and label have different shape.
    """
    className_to_id = {}

    for k,v in id_to_className.items():
        className_to_id[v] = int(k)


    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)

    if train_data:
        dataset = 'train'
    else:
        dataset = 'trainval'

    sys.stdout.write('Processing ' + dataset)
    image_list, label_list = filepath
    num_images = len(image_list)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = build_data.ImageReader(image_format, channels=3)

    if job_type == 'Segmentation':
        label_reader = build_data.ImageReader(label_format, channels=1)

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(
            outputdir,
            '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num_images, shard_id))
                sys.stdout.flush()
                # Read the image.
                image_filename = image_list[i]
                image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
                height, width = image_reader.read_image_dims(image_data)

                if job_type =='Classification':
                    cls_label_name = label_list[i]
                    cls_label = className_to_id[cls_label_name]
                    example = build_data.image_class_to_tfexample(
                        image_data, os.path.splitext(image_filename.split('\\')[-1])[0], height, width, cls_label)
                elif job_type == 'Segmentation':
                # Read the semantic segmentation annotation.
                    seg_filename = label_list[i]
                    seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
                    seg_height, seg_width = label_reader.read_image_dims(seg_data)
                    if height != seg_height or width != seg_width:
                        raise RuntimeError(
                            'Segmentation JOB Error, Shape mismatched between image and label.')
                    # Convert to tf example.
                    example = build_data.image_seg_to_tfexample(
                        image_data, os.path.splitext(image_filename.split('\\')[-1])[0], height, width, seg_data)
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()
