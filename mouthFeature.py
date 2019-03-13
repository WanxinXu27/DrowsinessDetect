import cv2
import dlib
import numpy as np
import os
import pandas as pd


def get_mouth_feature(path):
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    detector = dlib.get_frontal_face_detector()


    def get_landmarks(im):
        rects = detector(im, 1)
        if len(rects) > 1:
            return "error"
        if len(rects) == 0:
            return "error"
        return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


    def annotate_landmarks(im, landmarks):
        im = im.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
        return im


    def top_lip(landmarks):
        top_lip_pts = []
        for i in range(50, 53):
            top_lip_pts.append(landmarks[i])
        for i in range(61, 64):
            top_lip_pts.append(landmarks[i])
        # top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
        top_lip_mean = np.mean(top_lip_pts, axis=0)
        return int(top_lip_mean[:, 1])


    def bottom_lip(landmarks):
        bottom_lip_pts = []
        for i in range(65, 68):
            bottom_lip_pts.append(landmarks[i])
        for i in range(56, 59):
            bottom_lip_pts.append(landmarks[i])
        # bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
        bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
        return int(bottom_lip_mean[:, 1])


    def mouth_open(image):
        landmarks = get_landmarks(image)

        if landmarks == "error":
            return image, 0, 0

        image_with_landmarks = annotate_landmarks(image, landmarks)
        top_lip_center = top_lip(landmarks)
        bottom_lip_center = bottom_lip(landmarks)
        lip_distance = abs(top_lip_center - bottom_lip_center)
        length_of_lip = landmarks[64] - landmarks[60]
        return image_with_landmarks, lip_distance, length_of_lip.item(0)


    print("[INFO] starting video file: " + path)
    cap = cv2.VideoCapture(path)
    # cap = cv2.VideoCapture(0)

    # ratio = []
    # yawn_count = 0

    yawns = 0
    yawn_status = False
    count = 0
    YAWN_FRAMES = 48
    while True:
        prev_yawn_status = yawn_status
        ret, frame = cap.read()
        if frame is None:
            break

        image_landmarks, lip_distance, length_lip = mouth_open(frame)
        # ratio.append(str(lip_distance) + '\t' + str(length_lip))
        if lip_distance > 0.5 * length_lip:
            yawn_status = True

            count += 1

            # cv2.putText(frame, "Subject is Yawning", (50, 450),
            #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            #
            # output_text = " Yawn Count: " + str(yawns + 1)
            #
            # cv2.putText(frame, output_text, (50, 50),
            #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)

        else:
            yawn_status = False
            if prev_yawn_status == True and count >= YAWN_FRAMES:
                yawns += 1
                # yawn_count = count
            count = 0

        cv2.imshow('Live Landmarks', image_landmarks)
        cv2.imshow('Yawn Detection', frame)
        #
        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break

    if prev_yawn_status == True and count >= YAWN_FRAMES:
        yawns += 1
        # yawn_count = count

    cap.release()
    cv2.destroyAllWindows()
    return yawns

    # return ratio, yawn_count


if __name__ == '__main__':
    # path = './data'
    # d = {'Video':[], 'Yawns' : []}
    #
    # totalFiles = len(os.listdir(path))
    # competed = 0
    #
    # for file in os.listdir(path):
    #     if file[0] != '.':
    #         data = get_mouth_feature(path + '/' + file)
    #         d['Video'].append(file)
    #         d['Yawns'].append(data)
    #     competed += 1
    #     print(str(competed) + '/' + str(totalFiles))
    #
    # df = pd.DataFrame(data=d)
    # df.to_csv('./output/mouthFeatures.csv')

    for file in ['IMG_8686_08.avi','IMG_8687_00.avi', 'IMG_8687_01.avi', 'IMG_8687_02.avi']:
        count = get_mouth_feature('./data/' + file)
    #     with open ('./output/yawn_threshold.txt', 'a') as f:
    #         # f.write('\n'.join(str(i) for i in ratio))
    #         f.write('count = ' + str(count))
    #         f.write('\n')
