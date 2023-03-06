import mediapipe as mp
import cv2
import numpy as np


def mediapipe_setup():
    #use to access different solutions for drawing
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose


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


    return results, image




def displayAngle(image, bodyPart, body_angle):
    cv2.putText(image, str(round(body_angle, 4)), 
            tuple(np.multiply(bodyPart, [640, 480]).astype(int)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)



# def workout(results, image):







def main():
    results, image = mediapipe_setup()
    print("done")



main()
