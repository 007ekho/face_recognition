# import cv2
# import dlib
# from scipy.spatial import distance
# import numpy as np

# # Constants
# EYE_AR_THRESH = 0.3
# EYE_AR_CONSEC_FRAMES = 3

# # Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Grab the indexes of the facial landmarks for the left and right eye, respectively
# (lStart, lEnd) = (42, 48)
# (rStart, rEnd) = (36, 42)

# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# def shape_to_np(shape, dtype="int"):
#     coords = np.zeros((68, 2), dtype=dtype)
#     for i in range(0, 68):
#         coords[i] = (shape.part(i).x, shape.part(i).y)
#     return coords

# def check_eye_blink(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rects = detector(gray, 0)

#     for rect in rects:
#         shape = predictor(gray, rect)
#         shape = shape_to_np(shape)

#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)

#         ear = (leftEAR + rightEAR) / 2.0

#         if ear < EYE_AR_THRESH:
#             return True
#     return False

# def detect_face(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     if len(faces) > 0:
#         return True
#     return False

# def main():
#     cap = cv2.VideoCapture(0)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         face_detected = detect_face(frame)
#         eye_blink_detected = check_eye_blink(frame)
        
#         if face_detected and eye_blink_detected:
#             print("Approval: Face and eye blink detected.")
#             cv2.putText(frame, "Approval: Face and eye blink detected.", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         else:
#             print("Disapproval: Check failed.")
#             cv2.putText(frame, "Disapproval: Check failed.", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         cv2.imshow("Frame", frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



# import streamlit as st
# import cv2
# import dlib
# from scipy.spatial import distance
# import numpy as np
# from PIL import Image

# # Constants
# EYE_AR_THRESH = 0.3
# EYE_AR_CONSEC_FRAMES = 3

# # Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Grab the indexes of the facial landmarks for the left and right eye, respectively
# (lStart, lEnd) = (42, 48)
# (rStart, rEnd) = (36, 42)

# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# def shape_to_np(shape, dtype="int"):
#     coords = np.zeros((68, 2), dtype=dtype)
#     for i in range(0, 68):
#         coords[i] = (shape.part(i).x, shape.part(i).y)
#     return coords

# def check_eye_blink(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rects = detector(gray, 0)

#     for rect in rects:
#         shape = predictor(gray, rect)
#         shape = shape_to_np(shape)

#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)

#         ear = (leftEAR + rightEAR) / 2.0

#         if ear < EYE_AR_THRESH:
#             return True
#     return False

# def detect_face(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     if len(faces) > 0:
#         return True
#     return False

# def main():
#     st.title("Facial Recognition with Eye Blink Detection")

#     # Create a placeholder for the video frame
#     frame_placeholder = st.empty()

#     # Open the webcam
#     cap = cv2.VideoCapture(0)

#     # Process the video stream
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         face_detected = detect_face(frame)
#         eye_blink_detected = check_eye_blink(frame)

#         if face_detected and eye_blink_detected:
#             message = "Approval: Face and eye blink detected."
#             color = (0, 255, 0)
#         else:
#             message = "Disapproval: Check failed."
#             color = (0, 0, 255)

#         # Display the message on the frame
#         cv2.putText(frame, message, (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         # Convert the frame to RGB format for Streamlit
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_image = Image.fromarray(frame_rgb)

#         # Display the frame in Streamlit
#         frame_placeholder.image(frame_image, use_column_width=True)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
###################################################################################################
###################################################################################################
#################################
#################################
###################################################################################################
###################################################################################################
#################################
#################################
#################################
# import streamlit as st
# import cv2
# import dlib
# from scipy.spatial import distance
# import numpy as np
# from PIL import Image

# # Constants
# EYE_AR_THRESH = 0.3
# EYE_AR_CONSEC_FRAMES = 3
# BRIGHTNESS_THRESH = 100  # Threshold for brightness
# ALIGNMENT_THRESH = 0.35  # Threshold for face alignment

# # Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Grab the indexes of the facial landmarks for the left and right eye, respectively
# (lStart, lEnd) = (42, 48)
# (rStart, rEnd) = (36, 42)

# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# def shape_to_np(shape, dtype="int"):
#     coords = np.zeros((68, 2), dtype=dtype)
#     for i in range(0, 68):
#         coords[i] = (shape.part(i).x, shape.part(i).y)
#     return coords

# def check_eye_blink(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rects = detector(gray, 0)

#     for rect in rects:
#         shape = predictor(gray, rect)
#         shape = shape_to_np(shape)

#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)

#         ear = (leftEAR + rightEAR) / 2.0

#         if ear < EYE_AR_THRESH:
#             return True
#     return False

# def detect_face(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     if len(faces) > 0:
#         return True
#     return False

# def check_brightness(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     mean_brightness = np.mean(gray)
#     return mean_brightness > BRIGHTNESS_THRESH

# def check_alignment(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rects = detector(gray, 0)

#     for rect in rects:
#         shape = predictor(gray, rect)
#         shape = shape_to_np(shape)

#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         nose = shape[27]  # Nose tip landmark

#         # Calculate the center of the face based on the nose position
#         face_center_x = rect.left() + (rect.right() - rect.left()) // 2
#         face_center_y = rect.top() + (rect.bottom() - rect.top()) // 2

#         # Calculate the distances from the face center to the eyes and nose
#         left_eye_dist = distance.euclidean((face_center_x, face_center_y), leftEye.mean(axis=0))
#         right_eye_dist = distance.euclidean((face_center_x, face_center_y), rightEye.mean(axis=0))
#         nose_dist = distance.euclidean((face_center_x, face_center_y), nose)

#         # Check if the distances are within a reasonable range
#         if abs(left_eye_dist - right_eye_dist) / nose_dist < ALIGNMENT_THRESH:
#             return True
#     return False

# def main():
#     st.title("Facial Recognition with Eye Blink Detection")

#     # Create a placeholder for the video frame
#     frame_placeholder = st.empty()

#     # Open the webcam
#     cap = cv2.VideoCapture(0)

#     # Process the video stream
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         well_lit = check_brightness(frame)
#         if not well_lit:
#             message = "Disapproval: Insufficient lighting."
#             color = (0, 0, 255)
#         else:
#             face_detected = detect_face(frame)
#             if face_detected:
#                 well_aligned = check_alignment(frame)
#                 if not well_aligned:
#                     message = "Disapproval: Please adjust your head."
#                     color = (0, 0, 255)
#                 else:
#                     eye_blink_detected = check_eye_blink(frame)
#                     if eye_blink_detected:
#                         message = "Approval: Face and eye blink detected."
#                         color = (0, 255, 0)
#                     else:
#                         message = "Disapproval: Eye blink not detected."
#                         color = (0, 0, 255)
#             else:
#                 message = "Disapproval: Face not detected."
#                 color = (0, 0, 255)

#         # Display the message on the frame
#         cv2.putText(frame, message, (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         # Convert the frame to RGB format for Streamlit
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_image = Image.fromarray(frame_rgb)

#         # Display the frame in Streamlit
#         frame_placeholder.image(frame_image, use_column_width=True)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()












# import streamlit as st
# import cv2
# import dlib
# from scipy.spatial import distance
# import numpy as np
# from PIL import Image

# # Constants
# EYE_AR_THRESH = 0.3
# EYE_AR_CONSEC_FRAMES = 3
# BRIGHTNESS_THRESH = 80  # Threshold for brightness
# ALIGNMENT_SPHERE_CENTER_X = 320  # Example sphere center x
# ALIGNMENT_SPHERE_CENTER_Y = 240  # Example sphere center y
# ALIGNMENT_SPHERE_RADIUS = 100    # Example sphere radius

# # Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Load pre-trained face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Grab the indexes of the facial landmarks for the left and right eye, respectively
# (lStart, lEnd) = (42, 48)
# (rStart, rEnd) = (36, 42)

# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# def shape_to_np(shape, dtype="int"):
#     coords = np.zeros((68, 2), dtype=dtype)
#     for i in range(0, 68):
#         coords[i] = (shape.part(i).x, shape.part(i).y)
#     return coords

# def check_eye_blink(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rects = detector(gray, 0)

#     for rect in rects:
#         shape = predictor(gray, rect)
#         shape = shape_to_np(shape)

#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)

#         ear = (leftEAR + rightEAR) / 2.0

#         if ear < EYE_AR_THRESH:
#             return True
#     return False

# def check_brightness(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     mean_brightness = np.mean(gray)
#     return mean_brightness > BRIGHTNESS_THRESH

# def verify_face_alignment(center_x, center_y, radius):
#     if abs(center_x - ALIGNMENT_SPHERE_CENTER_X) < ALIGNMENT_SPHERE_RADIUS // 2 and abs(center_y - ALIGNMENT_SPHERE_CENTER_Y) < ALIGNMENT_SPHERE_RADIUS // 2:
#         return True
#     return False

# def detect_and_verify_face(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         # Draw rectangle around the face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # Check if the face fits within the predefined sphere dimensions
#         center_x, center_y = x + w // 2, y + h // 2
#         radius = max(w, h) // 2

#         # Draw the sphere (circle) on the frame
#         cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)

#         # Verify if the face is well-aligned within the sphere
#         if verify_face_alignment(center_x, center_y, radius):
#             return True
#         else:
#             cv2.putText(frame, 'Align Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#     return False

# def main():
#     st.title("Facial Recognition with Eye Blink and Alignment Detection")

#     # Create a placeholder for the video frame
#     frame_placeholder = st.empty()

#     # Open the webcam
#     cap = cv2.VideoCapture(0)

#     # Process the video stream
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         well_lit = check_brightness(frame)
#         if not well_lit:
#             message = "Disapproval: Insufficient lighting."
#             color = (0, 0, 255)
#         else:
#             face_aligned = detect_and_verify_face(frame)
#             if face_aligned:
#                 eye_blink_detected = check_eye_blink(frame)
#                 if eye_blink_detected:
#                     message = "Approval: Face and eye blink detected."
#                     color = (0, 255, 0)
#                 else:
#                     message = "Disapproval: Eye blink not detected."
#                     color = (0, 0, 255)
#             else:
#                 message = "Disapproval: Please align your face."
#                 color = (0, 0, 255)

#         # Display the message on the frame
#         cv2.putText(frame, message, (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         # Convert the frame to RGB format for Streamlit
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_image = Image.fromarray(frame_rgb)

#         # Display the frame in Streamlit
#         frame_placeholder.image(frame_image, use_column_width=True)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()













# import streamlit as st
# import cv2
# import dlib
# from scipy.spatial import distance
# import numpy as np
# from PIL import Image

# # Constants
# EYE_AR_THRESH = 0.3
# EYE_AR_CONSEC_FRAMES = 3
# BRIGHTNESS_THRESH = 70  # Threshold for brightness
# ALIGNMENT_SPHERE_CENTER_X = 320  # Example sphere center x
# ALIGNMENT_SPHERE_CENTER_Y = 240  # Example sphere center y
# ALIGNMENT_SPHERE_RADIUS = 100    # Example sphere radius

# # Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Load pre-trained face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Grab the indexes of the facial landmarks for the left and right eye, respectively
# (lStart, lEnd) = (42, 48)
# (rStart, rEnd) = (36, 42)

# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# def shape_to_np(shape, dtype="int"):
#     coords = np.zeros((68, 2), dtype=dtype)
#     for i in range(0, 68):
#         coords[i] = (shape.part(i).x, shape.part(i).y)
#     return coords

# def check_eye_blink(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rects = detector(gray, 0)

#     for rect in rects:
#         shape = predictor(gray, rect)
#         shape = shape_to_np(shape)

#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)

#         ear = (leftEAR + rightEAR) / 2.0

#         if ear < EYE_AR_THRESH:
#             return True
#     return False

# def check_brightness(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     mean_brightness = np.mean(gray)
#     return mean_brightness > BRIGHTNESS_THRESH

# def verify_face_alignment(center_x, center_y, radius):
#     if abs(center_x - ALIGNMENT_SPHERE_CENTER_X) < ALIGNMENT_SPHERE_RADIUS // 2 and abs(center_y - ALIGNMENT_SPHERE_CENTER_Y) < ALIGNMENT_SPHERE_RADIUS // 2:
#         return True
#     return False

# def detect_and_verify_face(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         # Check if the face fits within the predefined sphere dimensions
#         center_x, center_y = x + w // 2, y + h // 2
#         radius = max(w, h) // 2

#         # Draw the sphere (circle) on the frame
#         cv2.circle(frame, (ALIGNMENT_SPHERE_CENTER_X, ALIGNMENT_SPHERE_CENTER_Y), ALIGNMENT_SPHERE_RADIUS, (0, 255, 0), 2)

#         # Verify if the face is well-aligned within the sphere
#         if verify_face_alignment(center_x, center_y, radius):
#             return True
#         else:
#             cv2.putText(frame, 'Align Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#     return False

# def main():
#     st.title("Facial Recognition with Eye Blink and Alignment Detection")

#     # Create a placeholder for the video frame
#     frame_placeholder = st.empty()

#     # Open the webcam
#     cap = cv2.VideoCapture(0)

#     # Process the video stream
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         well_lit = check_brightness(frame)
#         if not well_lit:
#             message = "Disapproval: Insufficient lighting."
#             color = (0, 0, 255)
#         else:
#             face_aligned = detect_and_verify_face(frame)
#             if face_aligned:
#                 eye_blink_detected = check_eye_blink(frame)
#                 if eye_blink_detected:
#                     message = "Approval: Face and eye blink detected."
#                     color = (0, 255, 0)
#                 else:
#                     message = "Disapproval: Eye blink not detected."
#                     color = (0, 0, 255)
#             else:
#                 message = "Disapproval: Please align your face."
#                 color = (0, 0, 255)

#         # Display the message on the frame
#         cv2.putText(frame, message, (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         # Convert the frame to RGB format for Streamlit
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_image = Image.fromarray(frame_rgb)

#         # Display the frame in Streamlit
#         frame_placeholder.image(frame_image, use_column_width=True)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



import cv2
import streamlit as st
from PIL import Image
import numpy as np
import dlib
from scipy.spatial import distance

# Constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
BRIGHTNESS_THRESH = 100  # Threshold for brightness
ALIGNMENT_THRESH = 0.35  # Threshold for face alignment

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def check_eye_blink(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            return True
    return False

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        return True
    return False

def check_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > BRIGHTNESS_THRESH

def check_alignment(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        nose = shape[27]  # Nose tip landmark

        # Calculate the center of the face based on the nose position
        face_center_x = rect.left() + (rect.right() - rect.left()) // 2
        face_center_y = rect.top() + (rect.bottom() - rect.top()) // 2

        # Calculate the distances from the face center to the eyes and nose
        left_eye_dist = distance.euclidean((face_center_x, face_center_y), leftEye.mean(axis=0))
        right_eye_dist = distance.euclidean((face_center_x, face_center_y), rightEye.mean(axis=0))
        nose_dist = distance.euclidean((face_center_x, face_center_y), nose)

        # Check if the distances are within a reasonable range
        if abs(left_eye_dist - right_eye_dist) / nose_dist < ALIGNMENT_THRESH:
            return True
    return False

def main():
    st.title("Facial Recognition with Eye Blink Detection")

    # Create a placeholder for the video frame
    frame_placeholder = st.empty()

    # Open the webcam
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    # Process the video stream
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        well_lit = check_brightness(frame)
        if not well_lit:
            message = "Disapproval: Insufficient lighting."
            color = (0, 0, 255)
        else:
            face_detected = detect_face(frame)
            if face_detected:
                well_aligned = check_alignment(frame)
                if not well_aligned:
                    message = "Disapproval: Please adjust your head."
                    color = (0, 0, 255)
                else:
                    eye_blink_detected = check_eye_blink(frame)
                    if eye_blink_detected:
                        message = "Approval: Face and eye blink detected."
                        color = (0, 255, 0)
                    else:
                        message = "Disapproval: Eye blink not detected."
                        color = (0, 0, 255)
            else:
                message = "Disapproval: Face not detected."
                color = (0, 0, 255)

        # Display the message on the frame
        cv2.putText(frame, message, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Convert the frame to RGB format for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)

        # Display the frame in Streamlit
        frame_placeholder.image(frame_image, use_column_width=True)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



