import cv2
import os
import imutils


def clip_video(inputPath, outputPath, file):
    # file = 'IMG_8685'
    # inputPath = './video/'
    # outputPath = './data/'
    cap = cv2.VideoCapture(inputPath + file)
    fps = 30.0
    size = (253, 450)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    count = 0
    NUM_FRAMES = 120

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if count % NUM_FRAMES == 0:
                out = cv2.VideoWriter(outputPath + file[:-4] + '_' + str(count // NUM_FRAMES) + '.avi', fourcc, fps, size)
            frame = imutils.resize(frame, width=450)
            frame = cv2.transpose(frame)
            count += 1
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    inputPath = './videos/'
    outputPath = './data/'
    for file in os.listdir(inputPath):
        print('Processing File ' + file)
        clip_video(inputPath, outputPath, file)
