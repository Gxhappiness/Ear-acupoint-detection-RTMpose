# -*- coding = utf-8 -*-
# @Time : 2023/6/6 10:16
# @Author : Happiness
# @Software : PyCharm



#耳朵穴位检测关键点检测





#####  导入工具包

import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import torch

import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

from mmdet.apis import inference_detector, init_detector

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)



#####  载入待遇测图像

img_path = 'data/test_ear/1.JPG'

###########   两阶段的关键点检测（先目标检测，再关键点检测）

######   构建目标检测模型



# RTMDet 耳朵目标检测
detector = init_detector(
    'data/rtmdet_tiny_ear.py',
    'checkpoint/rtmdet_tiny_ear_epoch_175.pth',
    device=device
)





#####构建关键点检测模型
pose_estimator = init_pose_estimator(
    'data/rtmpose-s-ear.py',
    'checkpoint/rtmpose-s-ear-300.pth',
    device=device,
    cfg_options={'model': {'test_cfg': {'output_heatmaps': True}}}
)



#####   预测——————目标检测

init_default_scope(detector.cfg.get('default_scope', 'mmdet'))

# 获取目标检测预测结果
detect_result = inference_detector(detector, img_path)

# 预测类别
detect_result.pred_instances.labels

# 置信度
detect_result.pred_instances.scores





#########   置信度阈值过滤，获得最终目标检测预测结果、、

# 置信度阈值
CONF_THRES = 0.5

pred_instance = detect_result.pred_instances.cpu().numpy()
bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > CONF_THRES)]
bboxes = bboxes[nms(bboxes, 0.3)][:, :4].astype('int')





####   预测-——————关键点

# 获取每个 bbox 的关键点预测结果
pose_results = inference_topdown(pose_estimator, img_path, bboxes)

# 把多个bbox的pose结果打包到一起
data_samples = merge_data_samples(pose_results)






##### 预测结果-关键点坐标
keypoints = data_samples.pred_instances.keypoints.astype('int')





#####   预测结果-关键点热力图

# 每一类关键点的预测热力图
data_samples.pred_fields.heatmaps.shape

kpt_idx = 1
heatmap = data_samples.pred_fields.heatmaps[kpt_idx,:,:]

# 索引为 idx 的关键点，在全图上的预测热力图
plt.imshow(heatmap)
plt.show()



#### opencv可视化

img_bgr = cv2.imread(img_path)


# 检测框的颜色
bbox_color = (150,0,0)
# 检测框的线宽
bbox_thickness = 20
# 关键点半径
kpt_radius = 70
# 连接线宽
skeleton_thickness = 30

# 耳朵穴位关键点检测数据集-元数据（直接从config配置文件中粘贴，在data里面）
dataset_info = {
    'dataset_name':'Ear210_Keypoint_Dataset_coco',
    'classes':'ear',
    'paper_info':{
        'author':'Tongji Zihao',
        'title':'Triangle Keypoints Detection',
        'container':'OpenMMLab',
        'year':'2023',
        'homepage':'https://space.bilibili.com/1900783'
    },
    'keypoint_info': {
        0: {'name': '肾上腺', 'id': 0, 'color': [101, 205, 228], 'type': '', 'swap': ''},
        1: {'name': '耳尖', 'id': 1, 'color': [240, 128, 128], 'type': '', 'swap': ''},
        2: {'name': '胃', 'id': 2, 'color': [154, 205, 50], 'type': '', 'swap': ''},
        3: {'name': '眼', 'id': 3, 'color': [34, 139, 34], 'type': '', 'swap': ''},
        4: {'name': '口', 'id': 4, 'color': [139, 0, 0], 'type': '', 'swap': ''},
        5: {'name': '肝', 'id': 5, 'color': [255, 165, 0], 'type': '', 'swap': ''},
        6: {'name': '对屏尖', 'id': 6, 'color': [255, 0, 255], 'type': '', 'swap': ''},
        7: {'name': '心', 'id': 7, 'color': [255, 255, 0], 'type': '', 'swap': ''},
        8: {'name': '肺', 'id': 8, 'color': [29, 123,243], 'type': '', 'swap': ''},
        9: {'name': '肺2', 'id': 9, 'color': [0, 255, 255], 'type': '', 'swap': ''},
        10: {'name': '膀胱', 'id': 10, 'color': [128, 0, 128], 'type': '', 'swap': ''},
        11: {'name': '脾', 'id': 11, 'color': [74, 181, 57], 'type': '', 'swap': ''},
        12: {'name': '角窝中', 'id': 12, 'color': [165, 42, 42], 'type': '', 'swap': ''},
        13: {'name': '神门', 'id': 13, 'color': [128, 128, 0], 'type': '', 'swap': ''},
        14: {'name': '肾', 'id': 14, 'color': [255, 0, 0], 'type': '', 'swap': ''},
        15: {'name': '耳门', 'id': 15, 'color': [34, 139, 34], 'type': '', 'swap': ''},
        16: {'name': '听宫', 'id': 16, 'color': [255, 129, 0], 'type': '', 'swap': ''},
        17: {'name': '听会', 'id': 17, 'color': [70, 130, 180], 'type': '', 'swap': ''},
        18: {'name': '肩', 'id': 18, 'color': [63, 103,165], 'type': '', 'swap': ''},
        19: {'name': '扁桃体', 'id': 19, 'color': [66, 77, 229], 'type': '', 'swap': ''},
        20: {'name': '腰骶椎', 'id': 20, 'color': [255, 105, 180], 'type': '', 'swap': ''}
    },
    'skeleton_info': {
        0: {'link':('眼','扁桃体'),'id': 0,'color': [100,150,200]},
        1: {'link':('耳门','听宫'),'id': 1,'color': [200,100,150]},
        2: {'link':('听宫','听会'),'id': 2,'color': [150,120,100]},
        3: {'link':('耳门','听会'),'id': 3,'color': [66,77,229]}
    },
    'joint_weights':[1.0] * 21,
    'sigmas':[0.025] * 21
}

# 关键点类别和关键点ID的映射字典
label2id = {}
for each in dataset_info['keypoint_info'].items():
    label2id[each[1]['name']] = each[0]

for bbox_idx, bbox in enumerate(bboxes):  # 遍历每个检测框

    # 画框
    img_bgr = cv2.rectangle(img_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, bbox_thickness)

    # 索引为 0 的框，每个关键点的坐标
    keypoints = data_samples.pred_instances.keypoints[bbox_idx, :, :].astype('int')

    # 画连线
    for skeleton_id, skeleton in dataset_info['skeleton_info'].items():  # 遍历每一种连接
        skeleton_color = skeleton['color']
        srt_kpt_id = label2id[skeleton['link'][0]]  # 起始点的类别 ID
        srt_kpt_xy = keypoints[srt_kpt_id]  # 起始点的 XY 坐标
        dst_kpt_id = label2id[skeleton['link'][1]]  # 终止点的类别 ID
        dst_kpt_xy = keypoints[dst_kpt_id]  # 终止点的 XY 坐标
        img_bgr = cv2.line(img_bgr, (srt_kpt_xy[0], srt_kpt_xy[1]), (dst_kpt_xy[0], dst_kpt_xy[1]),
                           color=skeleton_color, thickness=skeleton_thickness)

    # 画关键点
    for kpt_idx, kpt_xy in enumerate(keypoints):  # 遍历该检测框中的每一个关键点
        kpt_color = dataset_info['keypoint_info'][kpt_idx]['color']
        img_bgr = cv2.circle(img_bgr, (kpt_xy[0], kpt_xy[1]), kpt_radius, kpt_color, -1)

plt.imshow(img_bgr[:,:,::-1])
plt.show()

cv2.imwrite('outputs/ear1_opencv.jpg', img_bgr)




#####   MMPose官方可视化工具visualizer

# 半径
pose_estimator.cfg.visualizer.radius = 50
# 线宽
pose_estimator.cfg.visualizer.line_width = 20
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
# 元数据
visualizer.set_dataset_meta(pose_estimator.dataset_meta)


img = mmcv.imread(img_path)
img = mmcv.imconvert(img, 'bgr', 'rgb')

img_output = visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=True,
            draw_bbox=True,
            show=False,
            show_kpt_idx=True,
            wait_time=0,
            out_file='outputs/ear1_visualizer.jpg',
            kpt_thr=0.3
)


plt.figure(figsize=(10, 10))
plt.imshow(img_output)
plt.show()