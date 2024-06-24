from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import base64
import threading
import mediapipe as mp
import numpy as np

import Ori_cutface 
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import cutface_region
import region_predict
import predict_putcolor
import putcolor_onface
import time


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

cap = None
lock = threading.Lock()
original_photo = None  # 存储原始拍摄的照片

# 初始化MediaPipe面部偵測器和面部標誌模型
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

lip_u = [0, 37, 39, 185, 80, 81, 82, 13, 312, 311, 310, 409, 270, 269, 267]
lip_l = [14, 87, 178, 88, 95, 146, 91, 181, 84, 17, 314, 405, 321, 375, 324, 318, 402, 317]

##Ori_cutface.py
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

###cutface_region.py, predict_putcolor.py
# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# new_model = region_predict.load_model('saved_model')
#######################################################################




# Load image using OpenCV
silhouette =  [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
  ]


rightEyeUpper0 =  [246, 161, 160, 159, 158, 157, 173]
rightEyeUpper1 = [247, 30, 29, 27, 28, 56, 190]
rightEyeLower0 = [33, 7, 163, 144, 145, 153, 154, 155, 133]
rightEyeLower1 = [130, 25, 110, 24, 23, 22, 26, 112, 243]
rightEyeLower3 = [143, 111, 117, 118, 119, 120, 121, 128, 245]
rightEyebrowLower = [ 124, 46, 53, 52, 65, 193]

leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]
leftEyeUpper1 = [467, 260, 259, 257, 258, 286, 414]
leftEyeLower0 = [263, 249, 390, 373, 374, 380, 381, 382, 362]
leftEyeLower1 = [359, 255, 339, 254, 253, 252, 256, 341, 463]
leftEyeLower3 = [372, 340, 346, 347, 348, 349, 350, 357, 465]

rightEyeLower0.reverse()
rightEyeLower1.reverse()
rightEyeLower3.reverse()
leftEyeLower0.reverse()
leftEyeLower1.reverse()
leftEyeLower3.reverse()

leftEyebrowLower = [276, 283, 282, 295, 285]

#######################################################################



def apply_lipstick(image, landmarks, color, intensity):
    lips_points_u = np.array([landmarks[i] for i in lip_u])
    lips_points_l = np.array([landmarks[i] for i in lip_l])
    
    mask = np.zeros_like(image)
    # 調整顏色值以匹配 BGR 格式
    adjusted_color = (int(color[2] * intensity), int(color[1] * intensity), int(color[0] * intensity))
    cv2.fillPoly(mask, [lips_points_u], adjusted_color)
    cv2.fillPoly(mask, [lips_points_l], adjusted_color)
    result = cv2.addWeighted(image, 1, mask, 0.4, 0)
    return result

@socketio.on('request_frame')
def handle_request_frame():
    global cap
    with lock:
        if cap is None:
            emit('frame_error', {'error': 'Camera not started'})
            return

        ret, frame = cap.read()
        if not ret:
            emit('frame_error', {'error': 'Failed to capture frame'})
            return

        frame = cv2.flip(frame, 1)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_str = base64.b64encode(buffer).decode('utf-8')

        emit('video_frame', {'frame1': frame_str, 'frame2': frame_str})

@socketio.on('capture_photo')
def handle_capture_photo():
    global cap, original_photo
    with lock:
        if cap is None:
            emit('photo_error', {'error': 'Camera not started'})
            return

        ret, frame = cap.read()
        if not ret:
            emit('photo_error', {'error': 'Failed to capture photo'})
            return

        frame = cv2.flip(frame, 1)
        original_photo = frame.copy()  # 存储原始照片
        _, buffer = cv2.imencode('.jpg', frame)
        photo_str = base64.b64encode(buffer).decode('utf-8')

        emit('photo_taken', {'photo': photo_str})

@socketio.on('start_camera')
def handle_start_camera():
    global cap
    with lock:
        if cap is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap = None
                emit('camera_error', {'error': 'Failed to start camera'})
                return
    print("Camera started")

@socketio.on('stop_camera')
def handle_stop_camera():
    global cap
    with lock:
        if cap is not None:
            cap.release()
            cap = None
    print("Camera stopped")

@app.route('/change_photo', methods=['POST'])
def change_photo():
    global original_photo
    data = request.get_json()
    lipColor = data['lipColor']
    lipIntensity = data['lipIntensity']
    eyeshadowColor = data['eyeshadowColor']
    eyeshadowIntense = data['eyeshadowIntense']
    eyeregionIntense = data['eyeregionIntense']
    
    if original_photo is None:
        return jsonify({'error': 'No original photo available'})

    # 确保每次基于原始照片进行处理
    photo_np = original_photo.copy()

    # 使用MediaPipe处理照片
    results = face_mesh.process(cv2.cvtColor(photo_np, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = photo_np.shape
            landmarks = [(int(point.x * width), int(point.y * height)) for point in face_landmarks.landmark]
            photo_np = apply_lipstick(photo_np, landmarks, lipColor, lipIntensity)
            cut_image = Ori_cutface.correct_and_crop_face(photo_np)
            cv2.imwrite('cut_image.jpg', cut_image)
            time.sleep(1)
            region_image=cutface_region.eye_region_generator(cut_image)
            cv2.imwrite('region_image.jpg', region_image)
            time.sleep(1)
            
            predict_image = region_predict.from_region_to_preict(region_image,'saved_model')
            cv2.imwrite('predict_image.jpg', predict_image)
            time.sleep(2)
            
            cut_image = cv2.imread('cut_image.jpg')
            cut_image1,predict_image1 = predict_putcolor.read_and_resize(cut_image,predict_image)
            cv2.imwrite('cut_image12.jpg', cut_image1)
            cv2.imwrite('predict_image12.jpg', predict_image1)
            cut_image1 = cv2.imread('cut_image12.jpg')
            predict_image1 = cv2.imread('predict_image12.jpg')
            final_image = predict_putcolor.put_color_on_face(cut_image1, predict_image1, eyeshadowColor, eyeshadowIntense, eyeregionIntense)
            cv2.imwrite('final_image.jpg', final_image)
            time.sleep(1)

            really_final=putcolor_onface.apply_mask_to_face(photo_np, final_image)
            cv2.imwrite('really_final.jpg', really_final)

        # 保存处理后的照片为JPG文件
    #cv2.imwrite('processed_photo.jpg', really_final)
    # print(eyeshadowIntense)
    # print(eyeshadowColor)
    # print(eyeregionIntense)
    _, buffer = cv2.imencode('.jpg', really_final) 
    photo_str = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'photo': photo_str})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)









