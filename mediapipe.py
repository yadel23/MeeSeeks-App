import mediapipe as mp
import cv2
import numpy as np

#use to access different solutions for drawing
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


# Curl counter variables
curl_counter = 0 
curl_stage = None

# pullup counter variables
pullup_counter = 0 
pullup_stage = None



# VIDEO FEED
#setting up video camera
cap = cv2.VideoCapture(0)
## Setup mediapipe instance
#Pose model, higher the number better the tracking
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        #ret is not used, frame is the image from the webcam
        ret, frame = cap.read()

        # Recolor image to RGB from BGR for mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False


        # Make detection storing it in results
        results = pose.process(image)

        # Recolor back to BGR from RGB because of opencv
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            #coordinates for Left curl
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate wrist_angle
            wrist_angle = calculate_angle(shoulder, elbow, wrist)
            
            # Visualize angle for left elbow
            cv2.putText(image, str(round(wrist_angle, 4)), 
                        tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Curl counter logic
            if wrist_angle > 160:
                curl_stage = "down"
            if wrist_angle < 30 and curl_stage =='down':
                curl_stage="up"
                curl_counter +=1
                print(curl_counter)



            #coordinates for pull ups
            # left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
            # mouth_left = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y


            #coordinates for pullups
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            # Calculate shoulder_angle
            shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)

            # Visualize angle for left shoulder
            cv2.putText(image, str(round(shoulder_angle, 4)), 
                        tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # pullup counter logic
            if shoudler_angle > 90:
                pullup_stage = "down"
            if shoulder_angle < 50 and pullup_stage == "down":
                pullup_stage = "up"
                pullup_counter += 1
                print(pullup_counter)

        except:
            pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(curl_counter), 
                    (5,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, curl_stage, 
                    (70,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)



        # Render pullup counter
        # Setup status box
        # cv2.rectangle(image, (0,0), (225,73), (0,117,16), -1)
        
        # # Rep data
        # cv2.putText(image, 'REPS', (15,12), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        # cv2.putText(image, str(pullup_counter), (10,60), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # # Stage data
        # cv2.putText(image, 'STAGE', (65,12), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        # cv2.putText(image, pullup_stage, (60,60), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)



        # Render detections
        #passing in: image, landmark_list, pose connections, landmark points drawings, connection lines drawings
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))                                 

        #outputs all x,y,z coordinates of each landmark(joint points)
        # print(results.pose_landmarks)
        #shows which body part are connected
        # mp_pose.Pose_CONNECTIONS

        #allows for visualization. passed the name and image=frame
        #cv2.flip(image, 1)
        cv2.imshow('Mediapipe Feed', image)

        #how to quit from the webcam view, 
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
#closes the window and releases the cam       
cap.release()
cv2.destroyAllWindows()