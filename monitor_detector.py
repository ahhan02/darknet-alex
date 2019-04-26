import sys, os
import os.path as osp
# os.chdir(osp.dirname(__file__))
# sys.path.append(os.path.join(os.getcwd(),'python/'))

import cv2
import glob
import tqdm
import shutil
import numpy as np
import argparse

classes = ['person']                                        #
#classes = ['hand']                                        #
#classes = ['head', 'hand']                                        #

def reverse_convert(box, h, w):
    xmin = int(box[0] - box[2]/2.0)
    xmax = int(box[0] + box[2]/2.0)
    ymin = int(box[1] - box[3]/2.0)
    ymax = int(box[1] + box[3]/2.0)

    if xmin < 0   : xmin = 0
    if xmax > w-1 : xmax = w - 1
    if ymin < 0   : ymin = 0
    if ymax > h-1 : ymax = h - 1
    return (xmin, ymin, xmax, ymax)


def draw_gt_rect(img, rsts):
    # rst cls, (xmin, ymin, xmax, ymax)
    for rst in rsts:
        cls = rst[0]
        bbox = rst[1]
        if cls in classes:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(img, str(cls), (
                int(bbox[0] + (bbox[2] - bbox[0]) * 0.5 - 10), int(bbox[1] + (bbox[3] - bbox[1]) * 0.5 - 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def draw_det_detect(img, rsts):
    # rst: cls, score. (cx, cy, w, h)
    for rst in rsts:
        cls = rst[0]
        conf = rst[1]
        bbox = reverse_convert(rst[2], img.shape[0], img.shape[1])
        if cls in classes:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(img, str(cls), (
                int(bbox[0] + (bbox[2] - bbox[0]) * 0.5 - 10), int(bbox[1] + (bbox[3] - bbox[1]) * 0.5 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, str(round(conf, 3)), (
                int(bbox[0]), int(bbox[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def maxIou(x1, y1, w1, h1, x2, y2, w2, h2):
		IOU = 0
		if ((abs(x1 - x2) < ((w1 + w2) / 2.0)) and (abs(y1 - y2) < ((h1 + h2) / 2.0))):
			left = max((x1 - (w1 / 2.0)), (x2 - (w2 / 2.0)))
			upper = max((y1 - (h1 / 2.0)), (y2 - (h2 / 2.0)))
			right = min((x1 + (w1 / 2.0)), (x2 + (w2 / 2.0)))
			bottom = min((y1 + (h1 / 2.0)), (y2 + (h2 / 2.0)))
			inter_w = abs(left - right)
			inter_h = abs(upper - bottom)
			inter_square = inter_w * inter_h
			area1 = w1 * h1
			area2 = w2 * h2
			iou1 = float(inter_square)/area1
			iou2 = float(inter_square)/area2
			# print("iou1:" , iou1)
			# print("iou2:" , iou2)
			if iou1 > iou2:
				return iou1
			else:
				return iou2
		return IOU


def readAnnotations(xml_path):
	import xml.etree.cElementTree as ET
	
	et = ET.parse(xml_path)
	element = et.getroot()
	element_objs = element.findall('object')
	
	results = []
	for element_obj in element_objs:
		result = []
		class_name = element_obj.find('name').text
		obj_bbox = element_obj.find('bndbox')
		x1 = int(round(float(obj_bbox.find('xmin').text)))
		y1 = int(round(float(obj_bbox.find('ymin').text)))
		x2 = int(round(float(obj_bbox.find('xmax').text)))
		y2 = int(round(float(obj_bbox.find('ymax').text)))
		
		result.append(class_name)
		#result.append(x1)
		#result.append(y1)
		#result.append(x2)
		#result.append(y2)
		result.append((x1, y1, x2, y2))
		
		results.append(result)
	return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--weights_file', type=str, required=True)
    parser.add_argument('--flag', type=str, choices=['0', '1'], required=True,)
    parser.add_argument('--img_path', type=str, required=False)
    parser.add_argument('--txt_path', type=str, required=False)
    parser.add_argument('--gpu_id', type=str, required=True)
    args = parser.parse_args()
    print(args)
	
    import darknet as dn
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id        # the gpu index begin from 0 now
    #dn.set_gpu(0)

    ################################################################################################

    #images = []
    #xmls = []
    if args.flag == '0':                                    # imagepath
        image_path = osp.join(args.img_path, 'image')
        xml_path = osp.join(args.img_path, 'xml')
        if not osp.exists(image_path) and not osp.exists(xml_path):
            print('imagepath does not exists')
            sys.exit(0)
        
        images = glob.glob(image_path + '/*.jpg')
        images.extend( glob.glob(image_path + '/*.jpeg') )

        xmls = glob.glob(xml_path + '/*.xml')

        #assert(len(images) == len(xmls), 'the length of images and xmls does not matched')      # simply check

    else:                                                   # txtpath
        txt_path = args.txt_path
        if not osp.exists(txt_path):
            print('txtpath does not exists')
            sys.exit(0)
        
        images = open(txt_path, 'r').readlines()
        images = [image.rstrip() for image in images]

        xmls = [image.replace('image', 'xml') for image in images]
        xmls = [xml.replace('jpg', 'xml') for xml in xmls]
        xmls = [xml.replace('jpeg', 'xml') for xml in xmls]

        #assert(len(images) == len(xmls), 'the length of images and xmls does not matched')      # simply check

    if len(images) == 0 or len(xmls) == 0:
        print('there are no images in given directory')
        sys.exit(0)

    ########################################################################################
    confidence = 0.25
    iou_threshold = 0.5
    #nms_threshold = 0.45
    nms_threshold = 0.6
    save_groundTruth = True # False

    result_path = "data/result_" + str(confidence)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    else:
        # os.mkdir(result_path)
        os.makedirs(result_path)

    no_detect_dir = os.path.join(result_path , "no_dets")
    if os.path.exists(no_detect_dir):
        shutil.rmtree(no_detect_dir)
    else:
        # os.mkdir(no_detect_dir) 
        os.makedirs(no_detect_dir)

    #
    images.sort()
    xmls.sort()
    for i in tqdm.tqdm(range(len(images))):
        img_name = images[i]
        xml_name = xmls[i]

        # print('img_name:' , img_name)

        if not os.path.exists(img_name):
            print('%s does not exist' % (img_name))
            continue

        img = cv2.imread(img_name)
        if type(img) == type(None):
            print('%s read failed' % (img_name))
            continue

        # draw gt boxes
        # groundtruth_results = readAnnotations(xml_name)
        # draw_gt_rect(img, groundtruth_results)

        # detect result
        # detect_results, _ = dn.performDetect(image=img_name, 
        detect_results, _ = dn.performDetect(image=img, 
            thresh=confidence, hier_thresh=iou_threshold, nms=nms_threshold,
            configPath=args.cfg_file, weightPath=args.weights_file, 
            metaPath=args.data_file, showImage=False)

        if len(detect_results) == 0:
            # print('%s:%s' % ("Do not detect any sku in the image" , os.path.basename(img_name)))
            # shutil.copyfile(xml_name , osp.join(no_detect_dir, os.path.basename(xml_name)))
            # shutil.copyfile(img_name , osp.join(no_detect_dir, os.path.basename(img_name)))
            cv2.imwrite( osp.join(no_detect_dir, osp.basename(img_name)), img )
            continue

        draw_det_detect(img, detect_results)

        # put it here
        # draw gt boxes
        groundtruth_results = readAnnotations(xml_name)
        draw_gt_rect(img, groundtruth_results)

        # for detect_result in detect_results:
        #     conf = detect_result[1]
        #     box = detect_result[2]

        cv2.imwrite( osp.join(result_path, osp.basename(img_name)), img)

