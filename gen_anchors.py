import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image
#def convert_coco_bbox(size, box):
#    """
#    Introduction
#    ------------
#        计算box的长宽和原始图像的长宽比值
#    Parameters
#    ----------
#        size: 原始图像大小
#        box: 标注box的信息
#    Returns
#        x, y, w, h 标注box和原始图像的比值
#    """
#    dw = 1. / size[0]
#    dh = 1. / size[1]
#    x = (box[0] + box[2]) / 2.0 - 1
#    y = (box[1] + box[3]) / 2.0 - 1
#    w = box[2]
#    h = box[3]
#    x = x * dw
#    w = w * dw
#    y = y * dh
#    h = h * dh
#    return x, y, w, h

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0-1
    y = (box[2] + box[3])/2.0-1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x, y, w, h]
def box_iou(boxes, clusters):
    """
    Introduction
    ------------
        计算每个box和聚类中心的距离值
    Parameters
    ----------
        boxes: 所有的box数据
        clusters: 聚类中心
    """
    box_num = boxes.shape[0]
    cluster_num = clusters.shape[0]
    box_area = boxes[:, 0] * boxes[:, 1]
    #每个box的面积重复9次，对应9个聚类中心
    box_area = box_area.repeat(cluster_num)
    box_area = np.reshape(box_area, [box_num, cluster_num])

    cluster_area = clusters[:, 0] * clusters[:, 1]
    cluster_area = np.tile(cluster_area, [1, box_num])
    cluster_area = np.reshape(cluster_area, [box_num, cluster_num])

    #这里计算两个矩形的iou，默认所有矩形的左上角坐标都是在原点，然后计算iou，因此只需取长宽最小值相乘就是重叠区域的面积
    boxes_width = np.reshape(boxes[:, 0].repeat(cluster_num), [box_num, cluster_num])
    clusters_width = np.reshape(np.tile(clusters[:, 0], [1, box_num]), [box_num, cluster_num])
    min_width = np.minimum(clusters_width, boxes_width)

    boxes_high = np.reshape(boxes[:, 1].repeat(cluster_num), [box_num, cluster_num])
    clusters_high = np.reshape(np.tile(clusters[:, 1], [1, box_num]), [box_num, cluster_num])
    min_high = np.minimum(clusters_high, boxes_high)

    iou = np.multiply(min_high, min_width) / (box_area + cluster_area - np.multiply(min_high, min_width))
    return iou


def avg_iou(boxes, clusters):
    """
    Introduction
    ------------
        计算所有box和聚类中心的最大iou均值作为准确率
    Parameters
    ----------
        boxes: 所有的box
        clusters: 聚类中心
    Returns
    -------
        accuracy: 准确率
    """
    return np.mean(np.max(box_iou(boxes, clusters), axis =1))


def Kmeans(boxes, cluster_num, iteration_cutoff = 25, function = np.median):
    """
    Introduction
    ------------
        根据所有box的长宽进行Kmeans聚类
    Parameters
    ----------
        boxes: 所有的box的长宽
        cluster_num: 聚类的数量
        iteration_cutoff: 当准确率不再降低多少轮停止迭代
        function: 聚类中心更新的方式
    Returns
    -------
        clusters: 聚类中心box的大小
    """
    boxes_num = boxes.shape[0]
    best_average_iou = 0
    best_avg_iou_iteration = 0
    best_clusters = []
    anchors = []
    np.random.seed()
    # 随机选择所有boxes中的box作为聚类中心
    clusters = boxes[np.random.choice(boxes_num, cluster_num, replace = False)]
    count = 0
    while True:
        distances = 1. - box_iou(boxes, clusters)
        boxes_iou = np.min(distances, axis=1)
        # 获取每个box距离哪个聚类中心最近
        current_box_cluster = np.argmin(distances, axis=1)
        average_iou = np.mean(1. - boxes_iou)
        if average_iou > best_average_iou:
            best_average_iou = average_iou
            best_clusters = clusters
            best_avg_iou_iteration = count
        # 通过function的方式更新聚类中心
        for cluster in range(cluster_num):
            clusters[cluster] = function(boxes[current_box_cluster == cluster], axis=0)
        if count >= best_avg_iou_iteration + iteration_cutoff:
            break
        print("Sum of all distances (cost) = {}".format(np.sum(boxes_iou)))
        print("iter: {} Accuracy: {:.2f}%".format(count, avg_iou(boxes, clusters) * 100))
        count += 1
    for cluster in best_clusters:
        anchors.append([round(cluster[0]*416 ), round(cluster[1]*416)])
    return anchors, best_average_iou

def process(cluster_num, iteration_cutoff = 25, function = np.median):
    """
    Introduction
    ------------
        主处理函数
    Parameters
    ----------
        dataFile: 数据集的标注文件
        cluster_num: 聚类中心数目
        iteration_cutoff: 当准确率不再降低多少轮停止迭代
        function: 聚类中心更新的方式
    """
    date_dir='E:/Python/tensorflow/YOLO/people count/yuncong_data/our/'
    with open('E:/Python/tensorflow/YOLO/people count/yuncong_data/our_train.txt') as f:
        boxes=[]
        GG = f.readlines()
        np.random.shuffle(GG)
        for line in (GG):
            line = line.split(' ')
            filename = line[0]
            image = Image.open(date_dir+filename)
            size_w,size_h=image.size
            dw=1/size_w
            dh=1/size_h 
            if filename[-1] == '\n':
                filename = filename[:-1]      
            num_people=int((len(line)-2)/5)
            for i in range(num_people):
                bb=(int(line[i*5+5])*dw,int(line[i*5+6])*dh)
                boxes.append(bb)
    boxes=np.array(boxes)
    last_best_iou = 0
    last_anchors = []
    box_w = boxes[:10000, 0]
    box_h = boxes[:10000, 1]
    plt.scatter(box_h, box_w, c = 'r')
    anchors = Kmeans(boxes, cluster_num, iteration_cutoff, function)
    anchors=np.array(anchors[0])
    plt.scatter(anchors[:,0], anchors[:, 1], c = 'b')
    plt.show()
    for _ in range(100):
        anchors, best_iou = Kmeans(boxes, cluster_num, iteration_cutoff, function)
        if best_iou > last_best_iou:
            last_anchors = anchors
            last_best_iou = best_iou
            print("anchors: {}, avg iou: {}".format(last_anchors, last_best_iou))
    print("final anchors: {}, avg iou: {}".format(last_anchors, last_best_iou))



if __name__ == '__main__':
    process(3)
