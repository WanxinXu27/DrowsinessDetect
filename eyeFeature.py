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


def get_eye_features(path):
    EYE_AR_THRESH = 0.3
    # EYE_AR_CONSEC_FRAMES = 48
    BLINK_FRAMES = 3
    totalBlink = 0
    COUNTER = 0
    # ALARM_ON = False

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    validFrame = 0
    avgClosureDegree = 0
    maxClosureFrames = 0


    print("[INFO] starting video file: " + path)
    cap = cv2.VideoCapture(path)
    totalFrame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    fvs = FileVideoStream(path).start()
    time.sleep(1.0)


    # loop over frames from the video file stream
    while fvs.more():
        frame = fvs.read()
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        if len(rects) > 0:
            validFrame += 1
            rect = rects[0]

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            avgClosureDegree += ear

            if ear < EYE_AR_THRESH:
                COUNTER += 1

            else:
                if COUNTER >= BLINK_FRAMES:
                    totalBlink += 1
                COUNTER = 0

            if COUNTER > maxClosureFrames:
                maxClosureFrames = COUNTER

        else:
            COUNTER = 0

        # cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    fvs.stop()

    # duration = videoProcessing.get_duration(fileName)
    validDuration = validFrame / 30
    duration = totalFrame / 30
    return {'BlinkRate': totalBlink/validDuration,
            'AvgClosureDegree': avgClosureDegree/validFrame,
            'MaxClosureFrames': maxClosureFrames,
            'ValidDuration': validDuration,
            'Blinks': totalBlink,
            'Duration': duration}


if __name__ == '__main__':
    path = './data'
    d = {'Video':[], 'BlinkRate': [], 'AvgClosureDegree': [], 'MaxClosureFrames': [], 'ValidDuration' : [], 'Blinks': [],
         'Duration' : []}

    totalFiles = len(os.listdir(path))
    competed = 0

    for file in os.listdir(path):
        if file[0] != '.':
            data = get_eye_features(path + '/' + file)
            d['Video'].append(file)
            d['BlinkRate'].append(data['BlinkRate'])
            d['AvgClosureDegree'].append(data['AvgClosureDegree'])
            d['MaxClosureFrames'].append(data['MaxClosureFrames'])
            d['ValidDuration'].append(data['ValidDuration'])
            d['Blinks'].append(data['Blinks'])
            d['Duration'].append(data['Duration'])
        competed += 1
        print(str(competed) + '/' + str(totalFiles))

    df = pd.DataFrame(data=d)
    df.to_csv('./output/eyeFeatures.csv')



