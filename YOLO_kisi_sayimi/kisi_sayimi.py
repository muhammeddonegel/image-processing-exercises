import cv2
from ultralytics import YOLO
import numpy as np

# load model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("video.mp4")

#counter and line location
line_position = [1000, 1200, 1400, 1600, 1800] # type of pixel
people_count = 0

positions = {}
crossed_lines = {}
already_counted = set()

def is_crossing_line(x, x_prev, line_x):
    return x_prev > line_x >= x

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(frame, persist = True, classes = [0]) # only "person" class 
    annotated_frame = results[0].plot()

    #draw line
    for line_x in line_position:
            cv2.line(annotated_frame, (line_x, 0), (line_x, frame.shape[0]), (0,255,255), 1)


    # take tracking info
    if results[0].boxes.id is not None:

        ids = results[0].boxes.id.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = box
            person_id = int(ids[i])
            center_x = int((x1 + x2) / 2)

            if person_id not in positions:
                positions[person_id] = center_x
                crossed_lines[person_id] = set()

            else:
                prev_x = positions[person_id]

                for line_x in line_position:

                    if (prev_x < line_x <= center_x) or (prev_x > line_x >= center_x):
                        crossed_lines[person_id].add(line_x)


                if len(crossed_lines[person_id]) >= 3 and person_id not in already_counted:
                    people_count += 1
                    already_counted.add(person_id)
                    print(f"Kişi geçti! Toplam: {people_count}")
                    
                positions[person_id] = center_x #update

            # number display
    cv2.putText(annotated_frame, f"Count: {people_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # show
    cv2.imshow("Kisi sayma sistemi", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
