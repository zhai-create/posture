import cv2
import time
import yaml
import mediapipe as mp
import os
import tensorflow as tf


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()

config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
session = tf.compat.v1.Session(config=config)



mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.8)

def process(image):
    start_time=time.time()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.face_landmarks,mp_holistic.FACEMESH_CONTOURS,mp_drawing.DrawingSpec((240,210,13),1,1),mp_drawing.DrawingSpec((240,210,13),1,1))
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,mp_drawing.DrawingSpec((0,0,255 ),1,1),mp_drawing.DrawingSpec((0,255,0),1,1))
    end_time=time.time()
    fps=int(1/(end_time-start_time))
    image = cv2.putText(image, 'fps:' + str(int(fps)), (25, 50 ), cv2.FONT_HERSHEY_SIMPLEX,
                      1.25 , (255, 0, 255), 2 )
    return image


# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(0)

# 打开cap
cap.open(0)

while cap.isOpened():
    # 获取画面
    success, frame = cap.read()
    if not success:
        break

    ## !!!处理帧函数
    frame = process(frame)

    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)

    if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
        break

# 关闭摄像头
cap.release()

# 关闭图像窗口
cv2.destroyAllWindows()












# cap = cv2.VideoCapture(0)
# time.sleep(2)
# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#     continue
#
#     cv2.imshow('MediaPipe Holistic', image)
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break
#     holistic.close()
#     cap.release()