import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh and drawing utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles





# Define the makeup effect function
def apply_eyeshade(image, mask, color,region_rate ,const):
    
    print(color)
    print(region_rate)
    print(const)

    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask1 = np.all(mask >255-region_rate*255/100, axis=-1)

    mask2 = image.copy()
    mask2[mask1] = gray_mask[mask1][:, np.newaxis] *color/255

# Parameters for the GuidedFilter
    radius = 10         # Radius of the guided filter
    eps = 0.1         # Regularization parameter (epsilon)

    guided_filter = cv2.ximgproc.createGuidedFilter(guide=gray_mask, radius=radius, eps=eps)
    filtered_mask = guided_filter.filter(mask2)
    result = cv2.addWeighted(image, const, filtered_mask, 1-const, 0)
    cv2.imwrite('777.jpg', result)
    return result

# Create trackbars to adjust lip color
def nothing(x):
    pass

# def create_trackbar():
#     cv2.namedWindow('Trackbars')
#     cv2.createTrackbar('R', 'Trackbars', 0, 255, nothing)
#     cv2.createTrackbar('G', 'Trackbars', 0, 255, nothing)
#     cv2.createTrackbar('B', 'Trackbars', 0, 255, nothing)
#     cv2.createTrackbar('level', 'Trackbars', 0, 100, nothing)
#     cv2.createTrackbar('region_rate', 'Trackbars', 0, 100, nothing)

# def get_trackbar():
#     r = cv2.getTrackbarPos('R', 'Trackbars')
#     g = cv2.getTrackbarPos('G', 'Trackbars')
#     b = cv2.getTrackbarPos('B', 'Trackbars')
#     const = cv2.getTrackbarPos('level', 'Trackbars')
#     region_rate = cv2.getTrackbarPos('region_rate', 'Trackbars')
#     return r,g,b,const,region_rate

def read_and_resize(cut_image,predict_image):
    # Read the image
    # image = cv2.imread(image_path)
    # mask = cv2.imread(mask_path)
    if cut_image is None:
        print(f"Failed to load image: image_path")
        exit()
    if predict_image is None:
        print(f"Failed to load image: mask_path")
        exit()
    cut_image1 = cv2.resize(cut_image, (128, 128), interpolation=cv2.INTER_LINEAR)
    predict_image1 = cv2.resize(predict_image, (128, 128))
    cv2.imwrite('cut_image1.jpg', cut_image1)
    cv2.imwrite('predict_image1.jpg', predict_image1)

    return cut_image1, predict_image1

# def show_img(output_image,r,g,b):
#     img = np.zeros((100, 100, 3), np.uint8) 
#     img[:] = [b,g,r]
#     cv2.imshow('Lipstick Application', output_image)
#     cv2.imshow('color plate',img)

def put_color_on_face(cut_image, predict_image, eyecolor, eyeshadow, eyeregion):
    # create_trackbar()
    # Detect face in the image
    results = face_mesh.process(cv2.cvtColor(cut_image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = cut_image.shape
            landmarks = [(int(point.x * width), int(point.y * height)) for point in face_landmarks.landmark]
    # while True:
        # Get the current positions of the trackbars
    r,g,b,const,region_rate = int(eyecolor[0]),int(eyecolor[1]), int(eyecolor[2]), eyeshadow, eyeregion
        
        # const compute method
    const = -(float(const)-50)/500+0.9

        # Apply the makeup effect with the current color
    if results.multi_face_landmarks:
        output_image = apply_eyeshade(cut_image.copy(),predict_image.copy() , (b, g, r),region_rate,const)
        
    else:
        output_image = cut_image.copy()
        
    output_image = cv2.resize(output_image, (512, 512))
        # Display the result
    #show_img(output_image,r,g,b)
        # Exit on 'esc' key press
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break
    #cv2.destroyAllWindows()
    #face_mesh.close()
    cv2.imwrite('fffi.jpg', output_image)
    return output_image

# if __name__ == '__main__':
#     # image_path = 'C:/Users/User/makeup_project_finaltest/cut_face'
#     # filename = 'cut.jpg'
#     # mask_path ='C:/Users/User/makeup_project_finaltest/outcome/'
#     # mask_filename = 'outcome.jpg'
#     cut_image = cv2.imread('cut_image.jpg')
#     predict_image = cv2.imread('predict_image.jpg')
#     # image, mask = read_and_resize(image_path+'/'+filename,mask_path+'/'+mask_filename)
#     image, mask = read_and_resize(cut_image,predict_image)
#     cv2.imwrite('123.jpg', image)
#     cv2.imwrite('456.jpg', mask)
#     eyecolor = [255, 0, 255]
#     eyeshadow=1.0
#     eyeregion=100
#     put_color_on_face(image, mask, eyecolor, eyeshadow, eyeregion)