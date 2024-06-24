import cv2
import numpy as np
import mediapipe as mp
# 使用 OpenCV 加載預訓練的人臉檢測器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def correct_and_crop_face(image):
    # 轉換成灰度圖像
    h, w, _ = image.shape
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 提取所有关键点的坐标
            x_min = w
            y_min = h
            x_max = y_max = 0
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min: x_min = x
                if y < y_min: y_min = y
                if x > x_max: x_max = x
                if y > y_max: y_max = y
            # 计算正方形区域
            face_width = x_max - x_min
            face_height = y_max - y_min
            face_size = max(face_width, face_height)
            center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
            # 确保正方形区域不超出边界
            left = max(center_x - face_size // 2, 0)
            right = min(center_x + face_size // 2, w)
            top = max(center_y - face_size // 2, 0)
            bottom = min(center_y + face_size // 2, h)
            # 裁剪人脸区域并调整大小
            face_crop = image[top:bottom, left:right]
            face_crop_resized = cv2.resize(face_crop, (256, 256))
            #=====================================================傳回256*256
            
            # 将反转后的图像调整回原始大小
            # face_crop_inverted = cv2.resize(face_crop_resized, (right-left, bottom-top))
            # # 将反转后的图像粘贴回原始图像中
            # image[top:bottom, left:right] = face_crop_inverted
            # # 绘制边界框
            # # cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    return face_crop_resized

# def correct_face_orientation(gray, img, faces):
#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
        
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         if len(eyes) >= 2:
#             # 取前兩個眼睛
#             eye1 = eyes[0]
#             eye2 = eyes[1]
            
#             if eye1[0] > eye2[0]:
#                 eye1, eye2 = eye2, eye1
            
#             left_eye_center = (int(eye1[0] + eye1[2] / 2), int(eye1[1] + eye1[3] / 2))
#             right_eye_center = (int(eye2[0] + eye2[2] / 2), int(eye2[1] + eye2[3] / 2))

#             dy = right_eye_center[1] - left_eye_center[1]
#             dx = right_eye_center[0] - left_eye_center[0]
#             angle = np.degrees(np.arctan2(dy, dx))

#             # 旋轉圖片
#             eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
#                            (left_eye_center[1] + right_eye_center[1]) // 2)
#             M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
#             rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

#             return rotated, (x, y, w, h)

#     return img, (x, y, w, h)

def read_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    return image


# 執行示例
# 讀取圖片
# input_image_path = 'path_to_your_image.jpg'  # 替換成您的圖片路徑
# img = cv2.imread(input_image_path)
# if img is None:
#     print(f"Error: Unable to read image at {input_image_path}")
# else:
#     process_single_image(img)


    # folder = 'C:/Users/User/makeup_project_finaltest/oringinal_photo'
    # filename = 'ori.jpg'
    # output_path='C:/Users/User/makeup_project_finaltest/cut_face'
    # outputfilename = "cut.jpg"
    # image = read_image(folder + '/'+ filename)
    
    # corrected_face = correct_and_crop_face(image)
    # cv2.imwrite(output_path+'/'+outputfilename,corrected_face)
