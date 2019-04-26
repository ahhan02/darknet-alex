#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES='0'

if [ $# != 1 ] ; then
  echo "Invalid input param"
  echo "  Usage ./automatic_annotation ANNOTATION_DIR"
  exit 0
fi

#python automatic_annotation_rgb.py \
#  --data_file cfg/rgb-0112.data \
#  --cfg_file cfg/yolov3-rgb-0112.cfg \
#  --weights_file backup-rgb-0112-v3-nocrop/yolov3-rgb-0112_final.weights \
#  --img_dir $1

#python automatic_annotation_realsense.py \
#  --data_file cfg/realsense-0119.data \
#  --cfg_file cfg/yolov3-realsense-0119.cfg \
#  --weights_file backup-realsense-0119-v3/yolov3-realsense-0119_final.weights \
#  --img_dir $1

#python automatic_annotation_person.py \
#  --data_file cfg/coco.data \
#  --cfg_file cfg/yolov3.cfg \
#  --weights_file yolov3.weights \
#  --img_dir $1

python automatic_annotation_person.py \
  --data_file cfg/person-0412.data \
  --cfg_file cfg/yolov3-tiny-person-0412.cfg \
  --weights_file backup-person-0412/yolov3-tiny-person-0412_final.weights \
  --img_dir $1
