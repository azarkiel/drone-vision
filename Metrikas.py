import logging
import cv2

def Average(lst):
    return sum(lst) / len(lst)

def printMetrics(img, frameActual, numFrames, fps, fps_avg, fps_max):
    tagsXPos = 30
    tagsYPos = 30

    cv2.putText(img, "Frame " + str(frameActual) + "/" + str(int(numFrames)) + " " + str(round(100 * frameActual / numFrames, 1)) + "%", (tagsXPos, tagsYPos), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(img, str(round(fps, 1)) + "FPS", (tagsXPos, tagsYPos + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(img, str(fps_avg) + "FPS_AVG", (tagsXPos, tagsYPos + 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(img, str(fps_max) + "FPS_MAX", (tagsXPos, tagsYPos + 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    return img

def prepareLog(filename, level):
    print("LOG="+filename+"\tLEVEL="+str(level))
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger