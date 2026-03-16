import cv2 as cv
import numpy as np


vid = cv.VideoCapture("bang_chuyen.mp4")

delay=400

while True:
    ret, frame = vid.read()
    if not ret:
        break
    if frame is not None:
        cv.imshow("video", frame)
    if cv.waitKey(1) == ord("q"):
        break

 #đếm số lượng hình tròn vượt qua line màu đỏ.   
cv.destroyAllWindows()