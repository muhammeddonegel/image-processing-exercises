import cv2

# All tracker methods in opencv
OPENCV_OBJECT_TRACKERS = {"csrt"       : cv2.legacy.TrackerCSRT_create,
                          "kcf"        : cv2.legacy.TrackerKCF_create,
                          "boosting"   : cv2.legacy.TrackerBoosting_create,
                          "mil"        : cv2.legacy.TrackerMIL_create,
                          "tld"        : cv2.legacy.TrackerTLD_create,
                          "medianflow" : cv2.legacy.TrackerMedianFlow_create,
                          "mosse"      : cv2.legacy.TrackerMOSSE_create
                          }

tracker_name = "kcf"

trackers  = cv2.legacy.MultiTracker_create()

video_path = "MOT17-04-DPM.mp4"
cap = cv2.VideoCapture(video_path)

fps = 30
f = 0
while True:
    
    ret, frame = cap.read()
    (H, W) = frame.shape[:2]
    frame = cv2.resize(frame, dsize = (960, 540))
    
    (success, boxes) = trackers.update(frame)
    
    info = [("Tracker: ", tracker_name), ("Succes: ", "YES" if success else "NO")]
    
    string_text = ""
    
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        string_text = string_text + text + " "
        
    cv2.putText(frame, text, (10, H - (i*20) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("t"):
        box = cv2.selectROI("Frame", frame, fromCenter = False)
        
        tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
        trackers.add(tracker, frame, box)
        
    elif key == ord("q"): break

    f += 1
    
cap.release()
cv2.destroyAllWindows()


















