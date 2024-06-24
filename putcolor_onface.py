#成功切出方塊+眼罩
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# 加载模型
# save_dir = 'C:/DL_model/eyemask_model'
# save_dir = 'densenet_model/saved_model'
# loaded_model = tf.keras.models.load_model(save_dir)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def apply_mask_to_face(frame,final_image):
    height, width, _ = frame.shape
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    image=frame
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            # 获取脸部区域
            x_min = width
            y_min = height
            x_max = y_max = 0

            for lm in face_landmarks.landmark:
                x, y = int(lm.x * width), int(lm.y * height)
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
            right = min(center_x + face_size // 2, width)
            top = max(center_y - face_size // 2, 0)
            bottom = min(center_y + face_size // 2, height)


            #decoded_imgs = cv2.imread(final_image)
            #========================================================
            # 在人脸框内显示 mask2 的效果，框外保持原始图像
            frame[top:bottom, left:right] = cv2.resize(final_image,(right-left,bottom-top))
            # mask_pic=cv2.resize(re_img,(right-left,bottom-top))
            # frame[top:bottom, left:right] = mask_pic

            # 绘制边界框
            # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    return frame

# if __name__ == '__main__':
#     folder = 'C:/Users/User/makeup_project_finaltest/oringinal_photo'
#     filename = 'ori.jpg'
#     final_filename = 'final_picture.jpg'
#     output_path='C:/Users/User/makeup_project_finaltest/'
#     output_filename= "stamp_back.jpg"

#     frame = cv2.imread(folder+'/'+filename)

#     result_frame = apply_mask_to_face(frame,final_filename) 
#     cv2.imwrite('eventually.jpg',result_frame)