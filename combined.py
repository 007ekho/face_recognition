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
##################################################################################################
##################################################################################################
################################
################################
##################################################################################################
##################################################################################################
################################
################################
################################
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

#         # Display the frame in Streamlit
#         st.image(frame_rgb, channels="RGB", use_column_width=True)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()

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













# import streamlit as st
# import cv2
# import dlib
# from scipy.spatial import distance
# import numpy as np

# # Constants
# EYE_AR_THRESH = 0.3
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

#     cap = None
#     camera_index = 0
#     while cap is None and camera_index < 10:  # Try up to 10 camera indices
#         cap = cv2.VideoCapture(camera_index)
#         if not cap.isOpened():
#             cap.release()
#             cap = None
#             camera_index += 1

#     if cap is None:
#         st.error("Failed to open webcam. Please check your camera connection.")
#         return

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

#         # Display the frame in Streamlit
#         st.image(frame_rgb, channels="RGB", use_column_width=True)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()

# if __name__ == "__main__":
#     main()






# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# import cv2
# import dlib
# from scipy.spatial import distance
# import numpy as np

# # Constants
# EYE_AR_THRESH = 0.3
# BRIGHTNESS_THRESH = 100  # Threshold for brightness
# ALIGNMENT_THRESH = 0.35  # Threshold for face alignment

# # Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Grab the indexes of the facial landmarks for the left and right eye, respectively
# (lStart, lEnd) = (42, 48)
# (rStart, rEnd) = (36, 42)

# class VideoTransformer(VideoTransformerBase):
#     def transform(self, frame):
#         img = frame.to_ndarray(format="bgr24")

#         # Process the frame for face detection and eye blink detection
#         well_lit = self.check_brightness(img)
#         if not well_lit:
#             message = "Disapproval: Insufficient lighting."
#             color = (0, 0, 255)
#         else:
#             face_detected = self.detect_face(img)
#             if face_detected:
#                 well_aligned = self.check_alignment(img)
#                 if not well_aligned:
#                     message = "Disapproval: Please adjust your head."
#                     color = (0, 0, 255)
#                 else:
#                     eye_blink_detected = self.check_eye_blink(img)
#                     if eye_blink_detected:
#                         message = "Approval: Face and eye blink detected."
#                         color = (0, 255, 0)
#                     else:
#                         message = "Disapproval: Eye blink not detected."
#                         color = (0, 0, 255)
#             else:
#                 message = "Disapproval: Face not detected."
#                 color = (0, 0, 255)

#         # Draw message on the frame
#         cv2.putText(img, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         return img

#     def check_eye_blink(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         rects = detector(gray, 0)

#         for rect in rects:
#             shape = predictor(gray, rect)
#             shape = self.shape_to_np(shape)

#             leftEye = shape[lStart:lEnd]
#             rightEye = shape[rStart:rEnd]
#             leftEAR = self.eye_aspect_ratio(leftEye)
#             rightEAR = self.eye_aspect_ratio(rightEye)

#             ear = (leftEAR + rightEAR) / 2.0

#             if ear < EYE_AR_THRESH:
#                 return True
#         return False

#     def detect_face(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray)

#         if len(faces) > 0:
#             return True
#         return False

#     def check_brightness(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         mean_brightness = np.mean(gray)
#         return mean_brightness > BRIGHTNESS_THRESH

#     def check_alignment(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         rects = detector(gray, 0)

#         for rect in rects:
#             shape = predictor(gray, rect)
#             shape = self.shape_to_np(shape)

#             leftEye = shape[lStart:lEnd]
#             rightEye = shape[rStart:rEnd]
#             nose = shape[27]  # Nose tip landmark

#             # Calculate the center of the face based on the nose position
#             face_center_x = rect.left() + (rect.right() - rect.left()) // 2
#             face_center_y = rect.top() + (rect.bottom() - rect.top()) // 2

#             # Calculate the distances from the face center to the eyes and nose
#             left_eye_dist = distance.euclidean((face_center_x, face_center_y), leftEye.mean(axis=0))
#             right_eye_dist = distance.euclidean((face_center_x, face_center_y), rightEye.mean(axis=0))
#             nose_dist = distance.euclidean((face_center_x, face_center_y), nose)

#             # Check if the distances are within a reasonable range
#             if abs(left_eye_dist - right_eye_dist) / nose_dist < ALIGNMENT_THRESH:
#                 return True
#         return False

#     def eye_aspect_ratio(self, eye):
#         A = distance.euclidean(eye[1], eye[5])
#         B = distance.euclidean(eye[2], eye[4])
#         C = distance.euclidean(eye[0], eye[3])
#         ear = (A + B) / (2.0 * C)
#         return ear

#     def shape_to_np(self, shape, dtype="int"):
#         coords = np.zeros((68, 2), dtype=dtype)
#         for i in range(0, 68):
#             coords[i] = (shape.part(i).x, shape.part(i).y)
#         return coords

# def main():
#     st.title("Facial Recognition with Eye Blink Detection")

#     # Face Analysis Application #
#     st.title("Real Time Face Analysis Application")
#     activities = ["Webcam Face Analysis", "About"]
#     choice = st.sidebar.selectbox("Select Activity", activities)

#     if choice == "Webcam Face Analysis":
#         st.header("Webcam Live Feed")
#         st.write("Click on start to use webcam and perform face analysis")

#         # Initialize webcam stream
#         webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

#     elif choice == "About":
#         st.subheader("About this app")
#         st.write("""
#                  Real-time face analysis application using OpenCV, dlib, and Streamlit.
#                  """)

#         st.markdown(
#             """
#             Developed by Mohammad Juned Khan
#             Email: Mohammad.juned.z.khan@gmail.com
#             [LinkedIn](https://www.linkedin.com/in/md-juned-khan)
#             """
#         )

# if __name__ == "__main__":
#     main()







# import streamlit as st
# import mediapipe as mp
# import cv2
# import numpy as np
# import tempfile
# import time
# from PIL import Image

# mp_drawing = mp.solutions.drawing_utils
# mp_face_mesh = mp.solutions.face_mesh

# DEMO_VIDEO = 'demo.mp4'
# DEMO_IMAGE = 'demo.jpg'

# st.title('Face Mesh Application using MediaPipe')

# st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
#         width: 350px;
#     }
#     [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
#         width: 350px;
#         margin-left: -350px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# st.sidebar.title('Face Mesh Application using MediaPipe')
# st.sidebar.subheader('Parameters')

# @st.cache()
# def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
#     # initialize the dimensions of the image to be resized and
#     # grab the image size
#     dim = None
#     (h, w) = image.shape[:2]

#     # if both the width and height are None, then return the
#     # original image
#     if width is None and height is None:
#         return image

#     # check to see if the width is None
#     if width is None:
#         # calculate the ratio of the height and construct the
#         # dimensions
#         r = height / float(h)
#         dim = (int(w * r), height)

#     # otherwise, the height is None
#     else:
#         # calculate the ratio of the width and construct the
#         # dimensions
#         r = width / float(w)
#         dim = (width, int(h * r))

#     # resize the image
#     resized = cv2.resize(image, dim, interpolation=inter)

#     # return the resized image
#     return resized

# app_mode = st.sidebar.selectbox('Choose the App mode',
# ['About App','Run on Image','Run on Video']
# )

# if app_mode =='About App':
#     st.markdown('In this application we are using **MediaPipe** for creating a Face Mesh. **StreamLit** is to create the Web Graphical User Interface (GUI) ')
#     st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
#         width: 400px;
#     }
#     [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
#         width: 400px;
#         margin-left: -400px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
#     )
#     st.video('https://www.youtube.com/watch?v=FMaNNXgB_5c&ab_channel=AugmentedStartups')

#     st.markdown('''
#           # About Me \n 
#             Hey this is ** Ritesh Kanjee ** from **Augmented Startups**. \n
           
#             If you are interested in building more Computer Vision apps like this one then visit the **Vision Store** at
#             www.augmentedstartups.info/visionstore \n
            
#             Also check us out on Social Media
#             - [YouTube](https://augmentedstartups.info/YouTube)
#             - [LinkedIn](https://augmentedstartups.info/LinkedIn)
#             - [Facebook](https://augmentedstartups.info/Facebook)
#             - [Discord](https://augmentedstartups.info/Discord)
        
#             If you are feeling generous you can buy me a **cup of  coffee ** from [HERE](https://augmentedstartups.info/ByMeACoffee)
             
#             ''')
# elif app_mode =='Run on Video':

#     st.set_option('deprecation.showfileUploaderEncoding', False)

#     use_webcam = st.sidebar.button('Use Webcam')
#     record = st.sidebar.checkbox("Record Video")
#     if record:
#         st.checkbox("Recording", value=True)

#     st.sidebar.markdown('---')
#     st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
#         width: 400px;
#     }
#     [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
#         width: 400px;
#         margin-left: -400px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
#         )
#     # max faces
#     max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
#     st.sidebar.markdown('---')
#     detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
#     tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

#     st.sidebar.markdown('---')

#     st.markdown(' ## Output')

#     stframe = st.empty()
#     video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
#     tfflie = tempfile.NamedTemporaryFile(delete=False)


#     if not video_file_buffer:
#         if use_webcam:
#             vid = cv2.VideoCapture(0)
#         else:
#             vid = cv2.VideoCapture(DEMO_VIDEO)
#             tfflie.name = DEMO_VIDEO
    
#     else:
#         tfflie.write(video_file_buffer.read())
#         vid = cv2.VideoCapture(tfflie.name)

#     width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps_input = int(vid.get(cv2.CAP_PROP_FPS))

#     #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
#     codec = cv2.VideoWriter_fourcc('V','P','0','9')
#     out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

#     st.sidebar.text('Input Video')
#     st.sidebar.video(tfflie.name)
#     fps = 0
#     i = 0
#     drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

#     kpi1, kpi2, kpi3 = st.columns(3)

#     with kpi1:
#         st.markdown("**FrameRate**")
#         kpi1_text = st.markdown("0")

#     with kpi2:
#         st.markdown("**Detected Faces**")
#         kpi2_text = st.markdown("0")

#     with kpi3:
#         st.markdown("**Image Width**")
#         kpi3_text = st.markdown("0")

#     st.markdown("<hr/>", unsafe_allow_html=True)

#     with mp_face_mesh.FaceMesh(
#     min_detection_confidence=detection_confidence,
#     min_tracking_confidence=tracking_confidence , 
#     max_num_faces = max_faces) as face_mesh:
#         prevTime = 0

#         while vid.isOpened():
#             i +=1
#             ret, frame = vid.read()
#             if not ret:
#                 continue

#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = face_mesh.process(frame)

#             frame.flags.writeable = True
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#             face_count = 0
#             if results.multi_face_landmarks:
#                 for face_landmarks in results.multi_face_landmarks:
#                     face_count += 1
#                     mp_drawing.draw_landmarks(
#                     image = frame,
#                     landmark_list=face_landmarks,
#                     connections=mp_face_mesh.FACE_CONNECTIONS,
#                     landmark_drawing_spec=drawing_spec,
#                     connection_drawing_spec=drawing_spec)
#             currTime = time.time()
#             fps = 1 / (currTime - prevTime)
#             prevTime = currTime
#             if record:
#                 #st.checkbox("Recording", value=True)
#                 out.write(frame)
#             #Dashboard
#             kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
#             kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
#             kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

#             frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
#             frame = image_resize(image = frame, width = 640)
#             stframe.image(frame,channels = 'BGR',use_column_width=True)

#     st.text('Video Processed')

#     output_video = open('output1.mp4','rb')
#     out_bytes = output_video.read()
#     st.video(out_bytes)

#     vid.release()
#     out. release()

# elif app_mode =='Run on Image':

#     drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

#     st.sidebar.markdown('---')

#     st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
#         width: 400px;
#     }
#     [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
#         width: 400px;
#         margin-left: -400px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
#     st.markdown("**Detected Faces**")
#     kpi1_text = st.markdown("0")
#     st.markdown('---')

#     max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
#     st.sidebar.markdown('---')
#     detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
#     st.sidebar.markdown('---')

#     img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

#     if img_file_buffer is not None:
#         image = np.array(Image.open(img_file_buffer))

#     else:
#         demo_image = DEMO_IMAGE
#         image = np.array(Image.open(demo_image))

#     st.sidebar.text('Original Image')
#     st.sidebar.image(image)
#     face_count = 0
#     # Dashboard
#     with mp_face_mesh.FaceMesh(
#     static_image_mode=True,
#     max_num_faces=max_faces,
#     min_detection_confidence=detection_confidence) as face_mesh:

#         results = face_mesh.process(image)
#         out_image = image.copy()

#         for face_landmarks in results.multi_face_landmarks:
#             face_count += 1

#             #print('face_landmarks:', face_landmarks)

#             mp_drawing.draw_landmarks(
#             image=out_image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACE_CONNECTIONS,
#             landmark_drawing_spec=drawing_spec,
#             connection_drawing_spec=drawing_spec)
#             kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
#         st.subheader('Output Image')
#         st.image(out_image,use_column_width= True)
# # Watch Tutorial at www.augmentedstartups.info/YouTube







import streamlit as st
import time
import cv2
import numpy as np

st.title("5 Seconds Video Recorder")

# Create a button to start the recording process
start_recording = st.button("Start Recording")

if start_recording:
    st.write("Recording for 5 seconds...")

    # Initialize video capture from camera
    cap = cv2.VideoCapture(0)

    # Get the start time
    start_time = time.time()

    frames = []
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            st.image(frame, channels="BGR")

    cap.release()

    # Create a video file from captured frames
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    st.write("Recording finished!")

    # Display the video file
    st.video('output.avi')


