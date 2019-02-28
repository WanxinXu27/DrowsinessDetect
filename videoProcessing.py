import cv2

cap = cv2.VideoCapture("./IMG_8686.MOV")
fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frameCount/fps

print('fps = ' + str(fps))
print('number of frames = ' + str(frameCount))
print('duration (S) = ' + str(duration))
minutes = int(duration/60)
seconds = duration%60
print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

cap.release()