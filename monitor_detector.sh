#!/usr/bin/env sh
# python shanghai_detector.py --cfg cfg/yolov3-tiny.cfg --model yolov3-tiny_180144.weights
#export CUDA_VISIBLE_DEVICES='1'

python monitor_detector.py \
  --data_file cfg/person-0412.data \
  --cfg_file cfg/yolov3-tiny-person-0412.cfg \
  --weights_file backup-person-0412/yolov3-tiny-person-0412_150000.weights \
  --txt_path /data/data/test_cases/valid.txt \
  --flag 1 \
  --gpu_id 1

#python monitor_detector.py \
#  --data_file cfg/rgb-0227.data \
#  --cfg_file cfg/yolov3-tiny-rgb-0227.cfg \
#  --weights_file backup-rgb-0227/yolov3-tiny-rgb-0227_final.weights \
#  --txt_path /home/xiaomeng.han/data/rgb-0227/valid.txt \
#  --flag 1 \
#  --gpu_id 6

#python monitor_detector.py \
#  --data_file cfg/realsense-0225.data \
#  --cfg_file cfg/yolov3-tiny-realsense-0225.cfg \
#  --weights_file backup-realsense-0225/yolov3-tiny-realsense-0225_final.weights \
#  --txt_path /home/xiaomeng.han/data/realsense-0225/valid.txt \
#  --flag 1 \
#  --gpu_id 6

#python3 write_csv.py \
#  --data_file cfg/rebar-0122.data \
#  --cfg_file backup-rebar-0122/yolov3-tiny-rebar-0122.cfg \
#  --weights_file backup-rebar-0122/yolov3-tiny-rebar-0122_final.weights \
#  --txt_path /home/xiaomeng.han/data/rebar-0122/testA.txt \
#  --flag 1 \
#  --gpu_id 6

#python3 write_csv.py \
#  --data_file cfg/rebar-0122.data \
#  --cfg_file backup-rebar-model-512/backup-rebar-0122-v3-all-rand/yolov3-rebar-0122.cfg \
#  --weights_file backup-rebar-model-512/backup-rebar-0122-v3-all-rand/yolov3-rebar-0122_final.weights \
#  --txt_path /home/xiaomeng.han/data/rebar-0122/testA.txt \
#  --flag 1 \
#  --gpu_id 1

#python3 write_csv.py \
#  --data_file cfg/rebar-0122.data \
#  --cfg_file backup-rebar-0122-v3-all-544/yolov3-rebar-0122.cfg \
#  --weights_file backup-rebar-0122-v3-all-544/yolov3-rebar-0122_final.weights \
#  --txt_path /home/xiaomeng.han/data/rebar-0122/testA.txt \
#  --flag 1 \
#  --gpu_id 6
