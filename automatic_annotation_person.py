import xml.etree.ElementTree as ET
from lxml import etree, objectify

import pickle
import os
import time
from os import listdir, getcwd
import os.path as osp
from os.path import join
#from darknet_annotation import performDetect
import darknet as dn
import glob
import shutil
import tqdm
import argparse


def reverse_convert(box, h, w):
    #print(box)
    #print(h, w)
    # (x, y, w, h) ==> (xmin, ymin, xmax, ymax)

    xmin = int(box[0] - box[2]/2.0)
    xmax = int(box[0] + box[2]/2.0)
    ymin = int(box[1] - box[3]/2.0)
    ymax = int(box[1] + box[3]/2.0)

    if xmin < 0   : xmin = 0
    if xmax > w-1 : xmax = w - 1
    if ymin < 0   : ymin = 0
    if ymax > h-1 : ymax = h - 1

    #print(xmin, ymin, xmax, ymax)
    return (xmin, ymin, xmax, ymax)


def main(data_file, cfg_file, weights_file, img_dir):
    imgs = os.listdir(img_dir)
    if imgs == []:
        return

    for img in tqdm.tqdm(imgs):
        # start = time.time()
        try:
            detections, shape = dn.performDetect(
                # imagePath=join(img_dir, img),
                image=join(img_dir, img),
                configPath=cfg_file,
                weightPath=weights_file,
                metaPath=data_file,
                showImage=False,)
            # print('Detection done in %f s' % (time.time() - start))
        except Exception as e:
            print('Warning! Detection failed. {}'.format(e))
            continue

        # print(detections)
        #if detections == []:
        #    nodet_dir = join(img_dir, 'no_det')
        #    if not osp.exists(nodet_dir):
        #        os.makedirs(nodet_dir)

        #    shutil.move(join(img_dir, img), join(nodet_dir, img))
        #    continue

        E = objectify.ElementMaker(annotate=False)
        anno_tree = E.anntation(
            E.folder(img_dir),
            E.filename(img),
            E.path(join(img_dir, img)),
            E.source(
                E.database('HHD')
            ),
            E.size(
                E.width(shape[0]),
                E.height(shape[1]),
                # E.depth(shape[2])
                E.depth(3)
            ),
            E.segmented(0),
        )

        # E2 = objectify.ElementMaker(annotate=False)
        for det in detections:
            name = det[0]

            if name != 'person':
               continue

            # (x, y, w, h)
            xmin, ymin, xmax, ymax = reverse_convert(det[2], shape[1], shape[0])
            # print(xmin, ymin, xmax, ymax)

            object_tree = E.object(
                E.name(name),
                E.pose('Unspecified'),
                E.truncated(0),
                E.difficult(0),
                E.bndbox(
                    E.xmin(xmin),
                    E.ymin(ymin),
                    E.xmax(xmax),
                    E.ymax(ymax)
                )
            )

            anno_tree.append(object_tree)

        etree.ElementTree(anno_tree).write(join(img_dir, osp.splitext(img)[0]+".xml"), pretty_print=True)

if __name__ == '__main__':
    # main()
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--data_file', default='cfg/coco.data', help='class name')
    parser.add_argument('--cfg_file', default='cfg/yolov3.cfg', help='cfg file')
    parser.add_argument('--weights_file', default='yolov3.weights', help='weights file')
    parser.add_argument('--img_dir', required=True, help='image directory')
    args = parser.parse_args()
    print(args)
    main(args.data_file, args.cfg_file, args.weights_file, args.img_dir)
