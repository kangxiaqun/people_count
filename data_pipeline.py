import tensorflow as tf
import matplotlib.pyplot as plt
import config as cfg
import numpy as np
from draw_boxes import draw_boxes
import cv2
def parser(example):
    features = {
                'xywhc': tf.FixedLenFeature([400], tf.float32),
                'img': tf.FixedLenFeature((), tf.string)}
    feats = tf.parse_single_example(example, features)
    coord = feats['xywhc']
    coord = tf.reshape(coord, [100, 4])

    img = tf.decode_raw(feats['img'], tf.float32)
    img = tf.reshape(img, [416, 416, 3])
#    img = tf.image.resize_images(img, [480, 640])
#

    img = tf.image.random_hue(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.minimum(img, 1.0)
    img = tf.maximum(img, 0.0)
    return img, coord


def data_pipeline(file_tfrecords, batch_size):
    dt = tf.data.TFRecordDataset(file_tfrecords)
    dt = dt.map(parser, num_parallel_calls=4)
    dt = dt.prefetch(batch_size)
    dt = dt.shuffle(buffer_size=20*batch_size)
    dt = dt.repeat()
    dt = dt.batch(batch_size)
    iterator = dt.make_one_shot_iterator()
    imgs, true_boxes = iterator.get_next()

    return imgs, true_boxes


if __name__ == '__main__':
    file_path = 'E:/Python/tensorflow/YOLO/people count/yuncong_data/our/trainval_2015.tfrecord'
    imgs, true_boxes = data_pipeline(file_path, cfg.batch_size)
    sess = tf.Session()
    imgs_, true_boxes_ = sess.run([imgs, true_boxes])
    imgs_=imgs_[0]
    true_boxes_=true_boxes_[0]*416
    num=len(true_boxes_)
    for i in range(num):
        x1=int(true_boxes_[i,0]-0.5*true_boxes_[i,2])
        y1=int(true_boxes_[i,1]-0.5*true_boxes_[i,3])
        x2=int(true_boxes_[i,0]+0.5*true_boxes_[i,2])
        y2=int(true_boxes_[i,1]+0.5*true_boxes_[i,3])
        cv2.rectangle(imgs_,(x1,y1),(x2,y2),(0,0,255),1)    
#    cv2.namedWindow("Image") 
    cv2.imshow("Image", imgs_) 
    cv2.waitKey (0)