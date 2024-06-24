import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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



# Replace with your image path

# Defien all original settings
# for i in os.listdir(image_path):

def eye_region_generator(image):
# def trimap_generator(image)
# def trimap_generator(image,image_path,i =''):
    righteyeout_position = []
    righteyein_position = []
    lefteyeout_position = []
    lefteyein_position = []
    righteyemargin_position = []
    lefteyemargin_position = []
# Convert the BGR image to RGB
    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to find facial landmarks
    results = face_mesh.process(rgb_image)

    # Check if landmarks were detected

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw facial landmarks on the image
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            # print(face_landmarks.landmark)
            # Print the landmark coordinates
            # for id, lm in enumerate(face_landmarks.landmark):
            #     x, y = int(lm.x * width*4), int(lm.y * height*4)
            #     if id in leftEyebrowLower+leftEyeLower3:
            #         cv2.putText(image,f'{id}',(x,y),cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
            #         cv2.circle(image,(x,y), 2, color=(0, 0, 255))

            
            
            # Right Eye out region
            for i in rightEyebrowLower+rightEyeLower3:
                lm = face_landmarks.landmark[i]
                x, y = int(lm.x * width), int(lm.y * height)
                righteyeout_position.append([x,y])
            righteyeout_position = np.array(righteyeout_position, dtype=np.int32).reshape((-1, 1, 2))
            
            # Right Eye margin region        
            for i in rightEyeUpper1+rightEyeLower1:
                lm = face_landmarks.landmark[i]
                x, y = int(lm.x * width), int(lm.y * height)
                righteyemargin_position.append([x,y])
            righteyemargin_position = np.array(righteyemargin_position, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(image, [righteyemargin_position],(255,255,255))  
            
            # Right Eye in region
            for i in rightEyeLower0+rightEyeUpper0:
                lm = face_landmarks.landmark[i]
                x, y = int(lm.x * width), int(lm.y * height)
                righteyein_position.append([x,y])
            righteyein_position = np.array(righteyein_position, dtype=np.int32).reshape((-1, 1, 2))        
            cv2.fillPoly(image, [righteyein_position],(0, 0, 0))      
                    
            # Left Eye out region
            for i in leftEyebrowLower+leftEyeLower3:
                lm = face_landmarks.landmark[i]
                x, y = int(lm.x * width), int(lm.y * height)
                lefteyeout_position.append([x,y])
            lefteyeout_position = np.array(lefteyeout_position, dtype=np.int32).reshape((-1, 1, 2))
            
            # Left Eye margin region        
            for i in leftEyeUpper1+leftEyeLower1:
                lm = face_landmarks.landmark[i]
                x, y = int(lm.x * width), int(lm.y * height)
                lefteyemargin_position.append([x,y])
            lefteyemargin_position = np.array(lefteyemargin_position, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(image, [lefteyemargin_position],(255,255,255))        
            
            # Left Eye in region
            for i in leftEyeLower0+leftEyeUpper0:
                lm = face_landmarks.landmark[i]
                x, y = int(lm.x * width), int(lm.y * height)
                lefteyein_position.append([x,y])
            lefteyein_position = np.array(lefteyein_position, dtype=np.int32).reshape((-1, 1, 2))        
            cv2.fillPoly(image, [lefteyein_position],(0, 0, 0))       
            
            # cv2.fillPoly(image, [rightEyebrowLower_position],(0, 0, 0))
            # Create a mask of the same size as the image, filled with zeros (black)
            
            mask = np.zeros_like(image)
            

            # # Fill the right eyebrow lower position on the mask with white color
            cv2.fillPoly(mask, [righteyeout_position], (255, 255, 255))
            cv2.fillPoly(mask, [lefteyeout_position], (255, 255, 255))
            # # Apply the inverted mask to the image
            image = cv2.bitwise_and(image, mask)        
                
            # Create mask2 to cover the image on color not black or white
            mask2 = np.zeros_like(image)
            # Create condition for non-black and non-white pixels
            condition = (image != [0, 0, 0])
            condition2 = (image == [255, 255, 255])
            condition = condition.all(axis=-1)
            condition2 = condition2.all(axis=-1)
            mask2[condition] = [255,255,255]
            mask2[condition2] = [255,255,255]
    face_mesh.close()
    return mask2



# if __name__ == '__main__':
#     image_path = '' 
#     index = 'cut_image.jpg'
#     output_image_path = os.path.join('C:\\Users\\User\\Desktop\\Tony\\', 'region_image.jpg')
#     image = cv2.imread('C:\\Users\\User\\Desktop\\Tony\\cut_image.jpg')

#     img = eye_region_generator(image)
#     cv2.imwrite('region_image', img)