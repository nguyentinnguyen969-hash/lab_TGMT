import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)  # hoặc URL ESP32-CAM

base_frame = None

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # 1. Chuyển xám
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (21, 21), 0)

    # 2. Khởi tạo background
    if base_frame is None:
        base_frame = gray.copy()
        continue

    # 3. Tính sai khác
    diff = cv.absdiff(base_frame, gray)

    # 4. Nhị phân hóa
    thresh = cv.threshold(diff, 30, 255, cv.THRESH_BINARY)[1]
    thresh = cv.dilate(thresh, None, iterations=2)

    # 5. Tìm contour
    contours, _ = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        if cv.contourArea(c) < 800:
            continue

        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # 6. Hiển thị
    cv.imshow("Frame", frame)
    cv.imshow("Threshold", thresh)

    key = cv.waitKey(1)
    if key == 27:  # ESC
        break

cam.release()
cv.destroyAllWindows()