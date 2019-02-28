# import the necessary packages
from imutils.video import FileVideoStream
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import videoProcessing
import os
import pandas as pd

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def get_eye_features(fileName):
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 48
    BLINK_FRAMES = 3
    totalBlink = 0
    COUNTER = 0
    ALARM_ON = False
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    numOfFrame = 0
    avgClosureDegree = 0
    maxClosureFrames = 0

    print("[INFO] starting video file thread...")
    fvs = FileVideoStream('./videos/' + fileName).start()
    time.sleep(1.0)


    # loop over frames from the video file stream
    while fvs.more():
        frame = fvs.read()
        if frame is None:
            break
        frame = imutils.resize(frame, width=450)
        frame = cv2.transpose(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        if len(rects) > 0:
            numOfFrame += 1
            rect = rects[0]

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            avgClosureDegree += ear

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                # if the eyes were closed for a sufficient number of
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if COUNTER >= BLINK_FRAMES:
                    totalBlink += 1
                COUNTER = 0
                ALARM_ON = False

            if COUNTER > maxClosureFrames:
                maxClosureFrames = COUNTER

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0

        # cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    fvs.stop()

    duration = videoProcessing.get_duration(fileName)
    return {'BlinkRate':totalBlink/duration,
            'AvgClosureDegree':avgClosureDegree/numOfFrame,
            'MaxClosureFrames': maxClosureFrames}
# python try.py --video IMG_8673.MOV --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

if __name__ == '__main__':
    path = os.getcwd()
    d = {'Video':[], 'BlinkRate': [], 'AvgClosureDegree': [], 'MaxClosureFrames': []}
    for file in os.listdir(path + '/videos'):
        data = get_eye_features(file)
        d['Video'].append(file)
        d['BlinkRate'].append(data['BlinkRate'])
        d['AvgClosureDegree'].append(data['AvgClosureDegree'])
        d['MaxClosureFrames'].append(data['MaxClosureFrames'])
    df = pd.DataFrame(data=d)
    writer = pd.ExcelWriter('./output/output.xlsx')
    df.to_excel(writer,'Sheet1',index=False)
    writer.save()



