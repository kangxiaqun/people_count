from yolo_top import yolov3
import numpy as np
import tensorflow as tf
import config as cfg
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
import os

test_dir='E:/Python/tensorflow/YOLO/people count/yuncong_data/'
f=open('E:/Python/tensorflow/YOLO/people count/yuncong_data/our/test1.txt','w')  
file='E:/Python/tensorflow/YOLO/people count/yuncong_data/merged_list.txt' 
line = open(file)
f_1=line.read()
img_name=f_1.split('\n')
num_img=len(img_name)
imgs_holder = tf.placeholder(tf.float32, shape=[1, 416, 416, 3])
istraining = tf.constant(False, tf.bool)
cfg.batch_size = 1
cfg.scratch = True
model = yolov3(imgs_holder, None, istraining)
img_hw = tf.placeholder(dtype=tf.float32, shape=[2])
boxes,scores = model.pedict(img_hw, iou_threshold=0.4)
saver = tf.train.Saver()
ckpt_dir = 'E:/Python/tensorflow/YOLO/people count/yuncong_data/our/ckpt/'

with tf.Session() as sess:
    saver.restore(sess, 'E:/Python/tensorflow/YOLO/people count/yuncong_data/our/ckpt/yolov3.ckpt-100000')
    for j in range(num_img):
        image_test = Image.open(test_dir+img_name[j]+'.jpg')
#        image_test = Image.open('E:/Python/tensorflow/YOLO/people count/yuncong_data/our/Part_B/test_data/IMG_128.jpg')
        resized_image = image_test.resize((416, 416), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
        a5=len(list(image_data.shape))
        if a5!=3:
            r=image_data
            g=image_data
            b=image_data
            image_data=cv2.merge([b,g,r])
        boxes_,scores_ = sess.run([boxes,scores],feed_dict={img_hw: [image_test.size[1], image_test.size[0]],
                                                        imgs_holder: np.reshape(image_data / 255, [1, 416, 416, 3])})              
#        image=np.array(image_test, dtype=np.float32)/255
#        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        num=len(boxes_)  
        f.write(img_name[j])
        f.write('\n')
        f.write(str(num))
        f.write('\n')
        for i in range(num):
            x=int((boxes_[i,3]-boxes_[i,1])/2+boxes_[i,1])
            y=int((boxes_[i,2]-boxes_[i,0])/2+boxes_[i,0])
            w=int(boxes_[i,3]-boxes_[i,1])
            h=int(boxes_[i,2]-boxes_[i,0])
            f.write(str(x))
            f.write(' ')
            f.write(str(y))
            f.write(' ')
            f.write(str(w))
            f.write(' ')
            f.write(str(h))
            f.write(' ')
            f.write(str(round(scores_[i],3)))
            f.write('\n')
#            cv2.rectangle(image,(boxes_[i,1],boxes_[i,0]),(boxes_[i,3],boxes_[i,2]),(0,0,255),2,2,0)                       
#        cv2.namedWindow("Image") 
#        cv2.imshow("Image", image) 
#        cv2.waitKey (0) 
    f.close()
