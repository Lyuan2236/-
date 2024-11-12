# 实现一种无监督或半监督的视频对象分割算法。
# 数据集使用 DAVIS-2016 或 DAVIS-2017
import os
import cv2
import numpy as np
from skimage import measure
from sklearn.metrics import f1_score
from PIL import Image


# 计算IoU（区域相似度）
def calculate_iou(pred, mask):
    pred = pred > 0.5  # 阈值化
    mask = mask > 0.5  # 阈值化
    intersection = np.logical_and(pred, mask).sum()
    union = np.logical_or(pred, mask).sum()
    iou = intersection / union if union != 0 else 0
    return iou


# 计算边界F1得分（轮廓精确度）
def calculate_boundary_f1(pred, mask, distance_threshold=1):
    pred_boundary = measure.find_contours(pred, 0.5)
    mask_boundary = measure.find_contours(mask, 0.5)

    # 检查是否检测到轮廓
    if len(pred_boundary) == 0 or len(mask_boundary) == 0:
        return None  # 如果没有检测到轮廓，跳过此帧

    pred_boundary_pixels = np.concatenate(pred_boundary, axis=0)
    mask_boundary_pixels = np.concatenate(mask_boundary, axis=0)

    # 计算预测边界到真实边界的距离
    precision = np.mean(
        [np.min(np.linalg.norm(mask_boundary_pixels - p, axis=1)) <= distance_threshold for p in pred_boundary_pixels])
    # 计算真实边界到预测边界的距离
    recall = np.mean(
        [np.min(np.linalg.norm(pred_boundary_pixels - m, axis=1)) <= distance_threshold for m in mask_boundary_pixels])

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


# DAVIS2017数据集加载
class DAVIS2017Dataset:
    def __init__(self, video_dir, annotations_dir, image_set_file, transform=None):
        self.video_dir = video_dir
        self.annotations_dir = annotations_dir
        self.transform = transform

        # 从 image_set_file 中读取视频文件名（train.txt）
        with open(image_set_file, 'r') as f:
            video_names = [line.strip() for line in f.readlines()]

        # 获取每个视频的帧文件路径
        self.video_files = []
        self.mask_files = []

        for video_name in video_names:
            # 每个视频文件夹名与视频名称一致，获取视频帧和掩码帧路径
            video_folder = os.path.join(video_dir, video_name)
            annotation_folder = os.path.join(annotations_dir, video_name)

            video_frames = sorted(
                [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.jpg')])
            mask_frames = sorted(
                [os.path.join(annotation_folder, f) for f in os.listdir(annotation_folder) if f.endswith('.png')])

            # 添加到总列表
            self.video_files.extend(video_frames)
            self.mask_files.extend(mask_frames)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        frame = Image.open(self.video_files[idx])  # 视频帧
        mask = Image.open(self.mask_files[idx]).convert('L')  # 对应的掩码，转换为灰度图

        if self.transform:
            frame = self.transform(frame)
            mask = self.transform(mask)

        return np.array(frame), np.array(mask)


# 半监督视频对象分割算法
def semi_supervised_video_segmentation(dataset):
    iou_scores = []
    f1_scores = []

    prev_frame_gray = None
    fgmask_subtractor = cv2.createBackgroundSubtractorMOG2()  # 创建背景建模对象

    for idx in range(len(dataset)):
        frame, mask = dataset[idx]

        # 将当前帧转换为灰度图
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # 计算光流（从第二帧开始计算）
        if prev_frame_gray is not None:
            # 检查当前帧与前一帧大小是否一致
            if prev_frame_gray.shape == curr_frame_gray.shape:
                flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, curr_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            else:
                print("Frame size mismatch between previous and current frame.")
                continue  # 跳过这一帧的处理，防止报错

            # 使用背景建模进行前景检测
            fgmask = fgmask_subtractor.apply(curr_frame_gray)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算去噪
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # 闭运算填补小空洞

            # 将fgmask归一化到0-1范围
            pred_mask = fgmask / 255.0
            pred_mask = np.clip(pred_mask, 0, 1)

            # 计算IoU和F1得分
            true_mask = mask / 255.0  # 真实标签
            iou = calculate_iou(pred_mask, true_mask)
            f1 = calculate_boundary_f1(pred_mask, true_mask)

            iou_scores.append(iou)
            f1_scores.append(f1)

            # 可视化分割结果
            pred_mask_color = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow('Segmentation', pred_mask_color)

        # 更新前一帧的灰度图
        prev_frame_gray = curr_frame_gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    avg_iou = np.mean(iou_scores) if iou_scores else 0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0
    print(f"Average IoU: {avg_iou:.4f}, Average Boundary F1: {avg_f1:.4f}")

    cv2.destroyAllWindows()


# 数据集路径
video_dir = 'E:/学科/大四上/数字视频处理/DAVIS-2017-trainval-480p/DAVIS/JPEGImages/480p'
annotations_dir = 'E:/学科/大四上/数字视频处理/DAVIS-2017-trainval-480p/DAVIS/Annotations/480p'
image_set_file = 'E:/学科/大四上/数字视频处理/DAVIS-2017-trainval-480p/DAVIS/ImageSets/2017/train.txt'

# 加载数据集
dataset = DAVIS2017Dataset(video_dir, annotations_dir, image_set_file)

# 执行半监督分割
semi_supervised_video_segmentation(dataset)
