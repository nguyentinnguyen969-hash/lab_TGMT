import cv2
from ultralytics import YOLO

VIDEO_PATH = "mvideo.mp4"
MODEL_PATH = "yolov8n.pt"

# car=2, motorcycle=3, truck=7
VEHICLE_CLASSES = [2,7]

COUNT_DIRECTION = "both"  # "both", "down", "up"
SHOW_ID = True
SHOW_CONF = True
COUNT_SIDE = "all"   # "left", "right", "all"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Không mở được video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

line_y = int(height * 0.7)
roi_x1, roi_y1 = 100, 100
roi_x2, roi_y2 = 600, 400
counted_ids = set()
total_count = 0
previous_centers = {}
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    #reset mỗi frame roi
    frame_count += 1
    roi_count = 0
    max_id = -1
    
    results = model.track(
        frame,
        persist=True,
        classes=VEHICLE_CLASSES,
        verbose=False
    )

    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2) #line
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
    

    if results[0].boxes is not None:
        for box in results[0].boxes:
            if box.id is None:
                continue
            track_id = int(box.id[0].item())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if track_id > max_id:
                max_id = track_id

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # ===== lọc bbox > 2000 =====
            # area = (x2 - x1) * (y2 - y1)
            # if area <= 2000:
            #     continue
            if COUNT_SIDE == "left" and cx >= width // 2:
                continue
            elif COUNT_SIDE == "right" and cx < width // 2:
                continue
             # ===== lọc roi =====
            if roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2:
                roi_count += 1
            
            class_name = model.names[cls_id]

            if class_name == "car":
                show_name = "oto"
            elif class_name == "truck":
                show_name = "truck"
            elif class_name == "motorcycle":
                show_name = "xemay"
            else:
                show_name = class_name
            
            if track_id % 2 == 0:
                color = (0, 255, 0)      # xanh lá (ID chẵn)
            else:
                color = (255, 0, 255)    # tím (ID lẻ) #đổi màu bbx

            if track_id in previous_centers:
                prev_cx, prev_cy = previous_centers[track_id]

                if COUNT_DIRECTION == "down":
                    crossed = (prev_cy < line_y <= cy)
                elif COUNT_DIRECTION == "up":
                    crossed = (prev_cy > line_y >= cy)
                else:
                    crossed = (prev_cy < line_y <= cy) or (prev_cy > line_y >= cy)

                if crossed and track_id not in counted_ids:
                    total_count += 1
                    counted_ids.add(track_id)
                    color = (255, 0, 0)

            previous_centers[track_id] = (cx, cy)

            # # ===== thay bbox bằng hình tròn =====
            # radius = max((x2 - x1), (y2 - y1)) // 2
            # cv2.circle(frame, (cx, cy), radius, color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # vẫn giữ tâm
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.circle(frame, (x1, y1), 5, color, -1)  # góc trên trái
            cv2.circle(frame, (x2, y1), 5, color, -1)  # góc trên phải
            cv2.circle(frame, (x1, y2), 5, color, -1)  # góc dưới trái
            cv2.circle(frame, (x2, y2), 5, color, -1)  # góc dưới phải
            label_parts = [show_name]

            if SHOW_ID:
                label_parts.append(f"ID:{track_id}")

            if SHOW_CONF:
                label_parts.append(f"{conf:.2f}")

            label = " | ".join(label_parts)

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )
        # hiển thị cảnh báo tắc đường
    if roi_count > 10:
        cv2.putText(
            frame,
            "TRAFFIC JAM",
            (470, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
    #     cv2.putText(
    #     frame,
    #     "ROI",
    #     (roi_x1, roi_y1 - 5),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.7,
    #     (255, 255, 0),
    #     2
    # )
    # nền đen cho ROI
    # cv2.rectangle(frame, (270, 10), (470, 60), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"ROI: {roi_count}",
        (280, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        3
    )
    # nền đen cho Total
    # cv2.rectangle(frame, (10, 10), (250, 60), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Count: {total_count}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        3
    )
        # hiển thị max ID
    # cv2.rectangle(frame, (270, 70), (470, 120), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Max ID: {max_id}",
        (280, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        3
    )
    # nền đen cho Frame
    cv2.putText(
        frame,
        f"Frame: {frame_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        3
    )

    cv2.imshow("Result", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()