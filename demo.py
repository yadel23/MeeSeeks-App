from pkg_resources import working_set
import mediapipe as mp
import cv2
import os
import numpy as np
import time
from flask import Flask, render_template, Response


cap = cv2.VideoCapture(0)



app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


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


filename = 'video.avi'
res = '720p'

# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


# workout_state = 2


# # Curl counter variables
# curl_counter = 0 
# curl_stage = None

# # pullup counter variables
# pullup_counter = 0 
# pullup_stage = None




def gen():
    # Curl counter variables
    curl_counter = 0 
    curl_stage = None

    # pullup counter variables
    pullup_counter = 0 
    pullup_stage = None

    # VIDEO FEED
    #setting up video camera
    # cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ## Setup mediapipe instance
    #Pose model, higher the number better the tracking
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: 
        video = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cap, res))

        # video = cv2.VideoWriter('webcamRecording.avi', fourcc, 25.0, (640, 480))
        # out = cv2.VideoWriter('output.avi',fourcc, 20.0,(int(cap.get(3)),int(cap.get(4))))

        while cap.isOpened():
            #ret is not used, frame is the image from the webcam
            
            ret, frame = cap.read()
    

            #if webcam is working write to videowriter
            if ret:
                video.write(frame)

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
                if shoulder_angle > 90:
                    pullup_stage = "down"
                if shoulder_angle < 50 and pullup_stage == "down":
                    pullup_stage = "up"
                    pullup_counter += 1
                    print(pullup_counter)

            except:
                pass

            print("in loop cur state: ", get_workout_state())

            if(get_workout_state() == 0):
                print("curl counter")
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
                
            elif(get_workout_state() == 1):
                print("pullup counter")
                # Render pullup counter
                # Setup status box
                cv2.rectangle(image, (0,0), (225,73), (0,117,16), -1)
                
                # Rep data
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(pullup_counter), (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, 'STAGE', (65,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, pullup_stage, (60,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)



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
            # cv2.imshow('Mediapipe Feed', image)
            ret, jpeg = cv2.imencode('.jpg', image)


            #how to quit from the webcam view, 
            if cv2.waitKey(10) & 0xFF == ord('q'):
                #closes the window and releases the cam       
                cap.release()
                cv2.destroyAllWindows()
                break
            else: 
                yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n') 


#display cam view in page
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


workout_state = {'state': 0}

def get_workout_state():
    return workout_state['state']


def set_workout_state(cur_state):
    workout_state['state'] = cur_state



#background process happening without any refreshing
@app.route('/background_process_pullups')
def background_process_pullups():
    print ("pullups")
    set_workout_state(1)
    return ("nothing")

@app.route('/background_process_curls')
def background_process_curls():
    print ("curls")
    set_workout_state(0)
    return ("nothing")

