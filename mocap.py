import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
# from pynput.keyboard import Controller
# keyboard = Controller()
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

iteration = 0

dfx = pd.DataFrame(columns = np.arange(33))
dfy = pd.DataFrame(columns = np.arange(33))
dfz = pd.DataFrame(columns = np.arange(33))


#VIDEO FEED
cap = cv2.VideoCapture(0)
# while cap.isOpened():
ret, frame = cap.read()
cv2.imshow('Mediapipe Feed', frame)

# if cv2.waitKey(10) & 0xFF == ord('q'):
#     break
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.75) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
    
        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            # print(landmarks)
            # if landmarks[mp_pose.PoseLandmark.NOSE.value].visibility > 0.95:
            #     print(landmarks[mp_pose.PoseLandmark.NOSE.value])
        except:
            pass
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                )               
        
        cv2.imshow('Mediapipe Feed', image)



        

        #start recording
        # if cv2.waitKey(10) & 0xFF == ord('k'):


        x_list = []
        y_list = []
        z_list = []
        vis_list = []
        for i in range(33):
            try:
                x_list.append(results.pose_landmarks.landmark[i].x)
                y_list.append(results.pose_landmarks.landmark[i].y)
                z_list.append(results.pose_landmarks.landmark[i].z)
                if results.pose_landmarks.landmark[i].visibility >= .80:
                    vis_list.append(results.pose_landmarks.landmark[i].visibility)
            except:
                x_list.append('NaN')
                y_list.append('NaN')
                z_list.append('NaN')
        if len(vis_list) >= 22:
            dfx.loc[len(dfx)] = x_list
            dfy.loc[len(dfy)] = y_list
            dfz.loc[len(dfz)] = z_list

            print('recording '+ str(iteration))


        if cv2.waitKey(10) & 0xFF == ord('q'):
            print('saved')
            dfx.to_json('mocap_x.json')
            dfy.to_json('mocap_y.json')
            dfz.to_json('mocap_z.json')
            break
        if cv2.waitKey(10) & 0xFF == ord('w'):
            print('custom save')
            file_name = input("Save as: ")
            os.chdir('capture_data')
            dfx.to_json(file_name +'x.json')
            dfy.to_json(file_name +'y.json')
            dfz.to_json(file_name +'z.json')

        iteration = iteration +1

cap.release()
cv2.destroyAllWindows()


