import cv2
import numpy as np

import dlib
from imutils import face_utils

# Read Image
im = cv2.imread("headPose.jpg");
cap = cv2.VideoCapture(0)

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

while True:
    _, im = cap.read()

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        #shape = face_utils.shape_to_np(shape)
        # Draw on our image, all the finded cordinate points (x,y) 
        cv2.circle(im, (shape.part(33).x,shape.part(33).x), 2, (0, 255, 0), -1)
        cv2.circle(im, (shape.part(8).x, shape.part(8).y), 2, (0, 255, 0), -1)
        cv2.circle(im, (shape.part(45).x, shape.part(45).y), 2, (0, 255, 0), -1)
        cv2.circle(im, (shape.part(36).x, shape.part(36).y), 2, (0, 255, 0), -1)
        cv2.circle(im, (shape.part(54).x, shape.part(54).y), 2, (0, 255, 0), -1)
        cv2.circle(im, (shape.part(48).x, shape.part(48).y), 2, (0, 255, 0), -1)
    
    
    size = im.shape
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                                (shape.part(33).x,shape.part(33).x),     # Nose tip
                                (shape.part(8).x, shape.part(8).y),      # Chin
                                (shape.part(45).x, shape.part(45).y),     # Left eye left corner
                                (shape.part(36).x, shape.part(36).y),     # Right eye right corne
                                (shape.part(54).x, shape.part(54).y),     # Left Mouth corner
                                (shape.part(48).x, shape.part(48).y)      # Right mouth corner
                            ], dtype="double")
     
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                             
                            ])
     
     
    # Camera internals
     
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
     
    print "Camera Matrix :\n {0}".format(camera_matrix)
     
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
     
    print "Rotation Vector:\n {0}".format(rotation_vector)
    print "Translation Vector:\n {0}".format(translation_vector)
     
     
    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
     
     
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
     
    ###for p in image_points:
        ###cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
     
     
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
     
    ###cv2.line(im, p1, p2, (255,0,0), 2)
     
    # Display image
    cv2.imshow("Output", im)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
