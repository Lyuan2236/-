import cv2
import numpy as np

video = cv2.VideoCapture("E:\学科\大四上\数字视频处理\\test.avi")

lk_params =  dict(winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))

feature_params = dict(maxCorners=100,qualityLevel=0.3,minDistance=7,blockSize=7)

ret,prev_frame = video.read()
prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

while True:
    ret,next_frame = video.read()
    if not ret:
        break

    next_gray = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, None, **lk_params)

    #Draw optional flow vectors  绘制运动矢量
    for i, (next_pt, prev_pt) in enumerate(zip(next_pts, prev_pts)):
        a, b = next_pt.ravel()
        c, d = prev_pt.ravel()
        # 绘制箭头
        cv2.arrowedLine(next_frame, (int(a),int(b)),(int(c),int(d)),(0,255,0),2)

    # 显示运动矢量图
    cv2.imshow("Optical Flow",next_frame)

    #Update the previous frame and points  # 更新前一帧和点
    prev_gray = next_gray.copy()
    prev_pts = next_pts.reshape(-1,1,2)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release