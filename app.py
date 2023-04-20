from pkg_resources import working_set
import mediapipe as mp
import cv2
import os
import numpy as np
import pandas as pd
import time
from flask import Flask, render_template, Response
from plotting import golfSwing
import time
from threading import Thread
import subprocess


cap = cv2.VideoCapture(0)


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




def gen():
    # Left Curl counter variables
    left_curl_counter = 0 
    left_curl_stage = None

    # Left Curl counter variables
    right_curl_counter = 0 
    right_curl_stage = None

    # pullup counter variables
    pullup_counter = 0 
    pullup_stage = None

    # squat counter variables
    squat_counter = 0 
    squat_stage = None

    # Lateral Raise variables
    lateral_raise_counter = 0
    lateral_raise_stage = None

    # Lateral Raise variables
    laying_leg_raise_counter = 0
    laying_leg_raise_stage = None



    iteration = 0

    dfx = pd.DataFrame(columns = np.arange(33))
    dfy = pd.DataFrame(columns = np.arange(33))
    dfz = pd.DataFrame(columns = np.arange(33))

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


            #default text top left box
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (300,117,16), -1)
            cv2.putText(image, 'REPS', (6,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

#**********************************************************************************************************************************************
                
                #coordinates for Left curl
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate left wrist_angle
                left_wrist_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                
                # left Curl counter logic
                if left_wrist_angle > 160:
                    left_curl_stage = "down"
                if left_wrist_angle < 30 and left_curl_stage =='down':
                    left_curl_stage="up"
                    left_curl_counter +=1
                    print(left_curl_counter)

#**********************************************************************************************************************************************


                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate right wrist_angle
                right_wrist_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                
                # right Curl counter logic
                if right_wrist_angle > 160:
                    right_curl_stage = "down"
                if right_wrist_angle < 30 and right_curl_stage =='down':
                    right_curl_stage="up"
                    right_curl_counter +=1
                    print(right_curl_counter)

#*********************************************************************************************************************************************

                #coordinates for pullups
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

                # Calculate shoulder_angle
                shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)

                # pullup counter logic
                if shoulder_angle > 90:
                    pullup_stage = "down"
                if shoulder_angle < 50 and pullup_stage == "down":
                    pullup_stage = "up"
                    pullup_counter += 1
                    print(pullup_counter)

#**********************************************************************************************************************************************

                #coordinates for squats
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate knee_angle
                knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                
                # squat counter logic
                if knee_angle >= 150:
                    squat_stage = "up"
                if knee_angle <= 90 and squat_stage == "up":
                    squat_stage = "down"
                    squat_counter += 1


#**********************************************************************************************************************************************
                ##Lateral workout left side
                # Calculate lateral_angle
                lateral_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                
                # squat counter logic
                if lateral_angle >= 75:
                    lateral_raise_stage = "up"
                if lateral_angle <= 15 and lateral_raise_stage == "up":
                    lateral_raise_stage = "down"
                    lateral_raise_counter += 1


#**********************************************************************************************************************************************
                ##leg raises workout left side
                #coordinates for leg raises
                leg_left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                leg_left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                leg_left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                # Calculate hip_angle
                leg_raise_angle = calculate_angle(leg_left_shoulder, leg_left_hip, leg_left_knee)
                
                # squat counter logic
                if leg_raise_angle >= 150:
                    laying_leg_raise_stage = "up"
                if leg_raise_angle <= 90 and laying_leg_raise_stage == "up":
                    squat_stage = "down"
                    laying_leg_raise_counter += 1


#**********************************************************************************************************************************************
#**********************************************************************************************************************************************

                #left curl workout
                if(get_workout_state() == 1):
                
                    ##reset other workouts
                    # Right Curl counter variables
                    right_curl_counter = 0 
                    right_curl_stage = None

                    # pullup counter variables
                    pullup_counter = 0 
                    pullup_stage = None

                    # squat counter variables
                    squat_counter = 0 
                    squat_stage = None

                    # Lateral Raise variables
                    lateral_raise_counter = 0
                    lateral_raise_stage = None

                    # Lateral Raise variables
                    laying_leg_raise_counter = 0
                    laying_leg_raise_stage = None

                    # Visualize angle for left elbow
                    cv2.putText(image, str(round(left_wrist_angle, 4)), 
                                tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    workout_text(image, left_curl_counter, left_curl_stage, 'Left Curl')

                    
    #**********************************************************************************************************************************************

                #pullup workout
                elif(get_workout_state() == 0):

                    #Reset other workouts
                    # Left Curl counter variables
                    left_curl_counter = 0 
                    left_curl_stage = None

                    # Left Curl counter variables
                    right_curl_counter = 0 
                    right_curl_stage = None

                    # squat counter variables
                    squat_counter = 0 
                    squat_stage = None

                    # Lateral Raise variables
                    lateral_raise_counter = 0
                    lateral_raise_stage = None

                    # Lateral Raise variables
                    laying_leg_raise_counter = 0
                    laying_leg_raise_stage = None

                    # Visualize angle for left shoulder
                    cv2.putText(image, str(round(shoulder_angle, 4)), 
                            tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    workout_text(image, pullup_counter, pullup_stage, 'Pullup')


    #**********************************************************************************************************************************************
                #Right curl workout
                elif(get_workout_state() == 2):

                    #Reset other workouts
                    # Left Curl counter variables
                    left_curl_counter = 0 
                    left_curl_stage = None

                    # pullup counter variables
                    pullup_counter = 0 
                    pullup_stage = None

                    # squat counter variables
                    squat_counter = 0 
                    squat_stage = None

                    # Lateral Raise variables
                    lateral_raise_counter = 0
                    lateral_raise_stage = None

                    # Lateral Raise variables
                    laying_leg_raise_counter = 0
                    laying_leg_raise_stage = None

                    # Visualize angle for right elbow
                    cv2.putText(image, str(round(right_wrist_angle, 4)), 
                                tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    workout_text(image, right_curl_counter, right_curl_stage, 'Right Curl')

                    
    #**********************************************************************************************************************************************
                #squat left side
                elif(get_workout_state() == 3):

                    #Reset other workouts
                    # Left Curl counter variables
                    left_curl_counter = 0 
                    left_curl_stage = None

                    # Left Curl counter variables
                    right_curl_counter = 0 
                    right_curl_stage = None

                    # pullup counter variables
                    pullup_counter = 0 
                    pullup_stage = None

                    # Lateral Raise variables
                    lateral_raise_counter = 0
                    lateral_raise_stage = None

                    # Lateral Raise variables
                    laying_leg_raise_counter = 0
                    laying_leg_raise_stage = None

                    # Visualize angle for left hip
                    cv2.putText(image, str(round(knee_angle, 4)), 
                                tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    workout_text(image, squat_counter, squat_stage, 'Squat')



#**********************************************************************************************************************************************

                #Lateral Raise
                elif(get_workout_state() == 4):

                    #Reset other workouts
                    # Left Curl counter variables
                    left_curl_counter = 0 
                    left_curl_stage = None

                    # Left Curl counter variables
                    right_curl_counter = 0 
                    right_curl_stage = None

                    # pullup counter variables
                    pullup_counter = 0 
                    pullup_stage = None

                    # squat counter variables
                    squat_counter = 0 
                    squat_stage = None

                    # Lateral Raise variables
                    laying_leg_raise_counter = 0
                    laying_leg_raise_stage = None

                    # Visualize angle for left shoulder
                    cv2.putText(image, str(round(shoulder_angle, 4)), 
                            tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    workout_text(image, lateral_raise_counter, lateral_raise_stage, 'Lat Raise')


#**********************************************************************************************************************************************

                #Leg Raise
                elif(get_workout_state() == 5):

                    #Reset other workouts
                    # Left Curl counter variables
                    left_curl_counter = 0 
                    left_curl_stage = None

                    # Left Curl counter variables
                    right_curl_counter = 0 
                    right_curl_stage = None

                    # pullup counter variables
                    pullup_counter = 0 
                    pullup_stage = None

                    # squat counter variables
                    squat_counter = 0 
                    squat_stage = None

                    # Lateral Raise variables
                    lateral_raise_counter = 0
                    lateral_raise_stage = None

                    # Visualize angle for left shoulder
                    cv2.putText(image, str(round(leg_raise_angle, 4)), 
                            tuple(np.multiply(leg_left_hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    workout_text(image, laying_leg_raise_counter, laying_leg_raise_stage, 'Leg Raise')


#**********************************************************************************************************************************************

            except:
                pass

#**********************************************************************************************************************************************
#**********************************************************************************************************************************************

            # Render detections
            #passing in: image, landmark_list, pose connections, landmark points drawings, connection lines drawings
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))                                 

            ret, jpeg = cv2.imencode('.jpg', image)


            # iteration = 0

            # dfx = pd.DataFrame(columns = np.arange(33))
            # dfy = pd.DataFrame(columns = np.arange(33))
            # dfz = pd.DataFrame(columns = np.arange(33))


            #list for all axis
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
            #does not record until 22 points are at least tracked
            if(get_workout_state() == "golf"):
                if len(vis_list) >= 22:
                    dfx.loc[len(dfx)] = x_list
                    dfy.loc[len(dfy)] = y_list
                    dfz.loc[len(dfz)] = z_list

                    print('recording '+ str(iteration))


            if(get_workout_state() == "golfSave"):
                # print('saved golf swing')
                # dfx.to_json('mocap_x.json')
                # dfy.to_json('mocap_y.json')
                # dfz.to_json('mocap_z.json')

                # time.sleep(10)
                # golfSwing()
                main()
                print("temp")
                # print(subprocess.run(["plotting.py", "arguments"], shell=True))



            #how to quit from the webcam view, 
            if cv2.waitKey(10) & 0xFF == ord('q'):
                #save recordings before closing
                print('saved')
                dfx.to_json('mocap_x.json')
                dfy.to_json('mocap_y.json')
                dfz.to_json('mocap_z.json')
                #closes the window and releases the cam       
                cap.release()
                cv2.destroyAllWindows()
                break
            else: 
                yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n') 

#**********************************************************************************************************************************************
#**********************************************************************************************************************************************


app = Flask(__name__)

#home screen
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index')
def index_redirect():
    return render_template('index.html')


@app.route('/recordingScreen')
def recordingScreen():
    return render_template('recordingScreen.html')



#display cam view in page
@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


#**********************************************************************************************************************************************
#**********************************************************************************************************************************************

workout_state = {'state': 0}

def get_workout_state():
    return workout_state['state']

def set_workout_state(cur_state):
    workout_state['state'] = cur_state


#background process happening without any refreshing
@app.route('/background_process_pullups')
def background_process_pullups():
    print ("pullups")
    set_workout_state(0)
    return ("nothing")

@app.route('/background_process_leftCurls')
def background_process_leftCurls():
    print ("left curls")
    set_workout_state(1)
    return ("nothing")

@app.route('/background_process_rightCurls')
def background_process_rightCurls():
    print ("right curls")
    set_workout_state(2)
    return ("nothing")

@app.route('/background_process_squat')
def background_process_squat():
    print ("squats")
    set_workout_state(3)
    return ("nothing")

@app.route('/background_process_lateral_raise')
def background_process_lateral_raise():
    print ("Lateral_raise")
    set_workout_state(4)
    return ("nothing")

@app.route('/background_process_leg_raise')
def background_process_leg_raise():
    print ("leg_raise")
    set_workout_state(5)
    return ("nothing")


@app.route('/background_process_golf')
def background_process_golf():
    print ("golf swing")
    set_workout_state("golf")
    return ("nothing")


@app.route('/background_process_golfSave')
def background_process_golfSave():
    print ("Saved golf swing")
    set_workout_state("golfSave")
    return ("nothing")


def workout_text(image, reps, stage, type):
    #reps data
    cv2.putText(image, str(reps), 
                    (5,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
                
    # Stage data
    cv2.putText(image, stage, 
                    (70,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
    #workout type
    cv2.putText(image, type, (140,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    


def main():
    # if(get_workout_state() != "golfSave"):
    #     app.run()
    # else:
    #     golfSwing()

    app.run(debug=False, host='0.0.0.0')

    # thread = Thread(target = app.run())
    # thread.start()
    # thread.join()
    
main()
