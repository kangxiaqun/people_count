from yolo_top import yolov3
import numpy as np
import tensorflow as tf
from data_pipeline import data_pipeline
import config as cfg
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
file_path = 'E:/Python/tensorflow/YOLO/people count/yuncong_data/our/trainval_2014.tfrecord'
log_dir='E:/Python/tensorflow/YOLO/people count/yuncong_data/our/log/'
imgs, true_boxes = data_pipeline(file_path, cfg.batch_size)

istraining = tf.constant(True, tf.bool)
model = yolov3(imgs, true_boxes, istraining)

with tf.name_scope('loss'):
    loss,AVG_IOU,coordinates_loss_sum,objectness_loss,no_objects_loss_mean = model.compute_loss()
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('avg', AVG_IOU)
    tf.summary.scalar('coord', coordinates_loss_sum)
    tf.summary.scalar('obj', objectness_loss)
    tf.summary.scalar('no_obj', no_objects_loss_mean)
global_step = tf.Variable(0, trainable=False)
#lr = tf.train.exponential_decay(0.0001, global_step=global_step, decay_steps=2e4, decay_rate=0.1)
lr = tf.train.piecewise_constant(global_step, [30000, 45000], [1e-4, 5e-5, 1e-5]) ##作用在不同步长时更改学习率
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
vars_det = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Head")
# for var in vars_det:
#     print(var)
with tf.control_dependencies(update_op):
#    train_op=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    train_op = optimizer.minimize(loss, global_step=global_step, var_list=vars_det)
summary_op = tf.summary.merge_all() 
saver = tf.train.Saver()
writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
ckpt_dir = 'E:/Python/tensorflow/YOLO/people count/yuncong_data/our/ckpt/'

gs = 0
batch_per_epoch = 10001
cfg.max_batches = int(batch_per_epoch * 10)
cfg.image_resized = 416
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
#    saver.restore(sess,'/home/jifuai/kangxq/YOLOv3/ckpt/yolov3.ckpt-50000')
    for i in range(cfg.max_batches):
        _, loss_,avg_iou_,coord_loss,obj_loss,no_obj_loss = sess.run([train_op,loss,AVG_IOU,coordinates_loss_sum,objectness_loss,no_objects_loss_mean])
        if(i % 100 == 0):
            print('step=%d,loss=%.5f,avg_iou=%.5f,coord_loss=%.5f,obj_loss=%.5f,no_obj_loss=%.5f'%(i,loss_,avg_iou_,coord_loss,obj_loss,no_obj_loss))
            #print('avg_iou', avg_iou_)
            summary_str = sess.run(summary_op)  
            writer.add_summary(summary_str, i) 
        if(i % 1000 == 0):
            saver.save(sess, ckpt_dir+'yolov3.ckpt', global_step=i, write_meta_graph=False)

