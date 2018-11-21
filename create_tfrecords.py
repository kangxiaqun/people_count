import numpy as np
import os
import tensorflow as tf
from PIL import Image
import sys
import cv2
sets = [('Part_B_train'),('our_train'),('UCSD_train'),('Mall_train')]##('our_train'),('UCSD_train'),('Mall_train')
date_dir='E:/Python/tensorflow/YOLO/people count/yuncong_data/our/'
tf_filename='trainval_2018.tfrecord'
max_boxes=100
tf_filename = os.path.join(date_dir,tf_filename)
writer = tf.python_io.TFRecordWriter(tf_filename)
for image_set in sets:
    with open('E:/Python/tensorflow/YOLO/people count/yuncong_data/%s.txt' % (image_set)) as f:
        j=0
        GG = f.readlines()
        np.random.shuffle(GG)
        for line in (GG):
            line = line.split(' ')
            line[-1]=line[-1].split('\n')[0]
            filename = line[0]
            if filename[-1] == '\n':
                filename = filename[:-1]
            if filename.split('.')[-1] !='jpg' :
                continue
            image = cv2.imread(date_dir+filename)
            if image.shape[2]!=3:
                continue
            image=cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(image,(416, 416), interpolation=cv2.INTER_CUBIC)
            image_data = np.array(resized_image, dtype='float32')/255
            img_raw = image_data.tobytes()
            size_w,size_h,_=image.shape
            dw=1/size_w
            dh=1/size_h        
            boxes = np.zeros((max_boxes, 4), dtype=np.float32)
            num_people=int((len(line)-2)/5)
            for i in range(num_people):
                boxes[i:i+1,0]=int(line[i*5+3])*dw+int(line[i*5+5])*dw*0.5
                boxes[i:i+1,1]=int(line[i*5+4])*dh+int(line[i*5+6])*dh*0.5
                boxes[i:i+1,2]=int(line[i*5+5])*dw
                boxes[i:i+1,3]=int(line[i*5+6])*dh
            boxes=np.array(boxes, dtype=np.float32).flatten().tolist()  
            j=j+1
            sys.stdout.write('\r>> Converting image %d/%d' % (j, len(GG)))
            sys.stdout.flush()  
            example = tf.train.Example(features=tf.train.Features(feature={
                'xywhc':
                        tf.train.Feature(float_list=tf.train.FloatList(value=boxes)),
                'img':
                        tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                }))
            writer.write(example.SerializeToString())
writer.close()
sys.stdout.write('\n')
sys.stdout.flush()
