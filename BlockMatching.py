import cv2
import numpy as np

def block_matching(ref_frame, curr_frame, block_size=8, search_range=4):
    h, w = ref_frame.shape
    motion_vectors = np.zeros((h // block_size, w // block_size, 2), dtype=np.float32)

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            ref_block = ref_frame[i:i + block_size, j:j + block_size]
            best_match = (0, 0)
            min_error = float('inf')

            for x in range(-search_range, search_range + 1):
                for y in range(-search_range, search_range + 1):
                    curr_x, curr_y = j + x, i + y
                    if 0 <= curr_x < w - block_size and 0 <= curr_y < h - block_size:
                        curr_block = curr_frame[curr_y:curr_y + block_size, curr_x:curr_x + block_size]

                        #使用绝对误差作为相似度量
                        # error = np.sum(np.abs(ref_block - curr_block))

                        #使用均方误差作为相似度量
                        error = np.sum((ref_block - curr_block) ** 2) / (block_size * block_size)

                        if error < min_error:
                            min_error = error
                            best_match = (curr_x, curr_y)

            motion_vectors[i // block_size, j // block_size] = (best_match[0] - j, best_match[1] - i)

    return motion_vectors

video = cv2.VideoCapture("E:\\学科\\大四上\\数字视频处理\\test.avi")

ret, ref_frame = video.read()
if not ret:
    print("无法读取视频帧")
    video.release()
    exit()

ref_frame_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, curr_frame = video.read()
    if not ret:
        break

    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    motion_vectors = block_matching(ref_frame_gray, curr_frame_gray)

    h, w = curr_frame_gray.shape

    # 在当前帧上叠加运动矢量
    curr_frame_with_vectors = curr_frame.copy()
    for i in range(motion_vectors.shape[0]):
        for j in range(motion_vectors.shape[1]):
            x, y = j * 8, i * 8
            mv = motion_vectors[i, j]
            start_point = (x + 4, y + 4)
            end_point = (int(start_point[0] + mv[0]), int(start_point[1] + mv[1]))

            if np.any(mv) and 0 <= end_point[0] < w and 0 <= end_point[1] < h:
                cv2.arrowedLine(curr_frame_with_vectors, start_point, end_point, (0, 255, 0), 1)

    # 调整图像尺寸
    scale_percent = 50  # 比例缩小50%
    width = int(ref_frame.shape[1] * scale_percent / 100)
    height = int(ref_frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    ref_frame_resized = cv2.resize(ref_frame, dim)
    curr_frame_resized = cv2.resize(curr_frame, dim)
    curr_frame_with_vectors_resized = cv2.resize(curr_frame_with_vectors, dim)

    # 拼接图像
    combined_image = np.hstack((ref_frame_resized, curr_frame_resized, curr_frame_with_vectors_resized))
    cv2.imshow("Reference Frame | Current Frame | Current Frame with Motion Vectors", combined_image)

    ref_frame_gray = curr_frame_gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()
