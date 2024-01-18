from engine.object_detection import ObjectDetection
import threading
import cv2

od = ObjectDetection("dnn_model/yolov8s.pt")
od.load_class_names("dnn_model/classes.txt")


def run_tracker_in_thread(filename, file_index):
    video = cv2.VideoCapture(filename) 
    while True:
        ret, frame = video.read()  
        if not ret:
            break

        bboxes, class_ids, scores = od.detect(frame, imgsz=640, conf=0.40, classes=[0])
        for bbox, class_id, score in zip(bboxes, class_ids, scores):
            (x, y, x2, y2) = bbox
            cv2.rectangle(frame, (x, y), (x2, y2), od.colors[class_id], 4)
            class_id = int(class_id)
            class_name = od.classes[class_id]
            cv2.putText(frame, class_name + " " + str(score), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, od.colors[class_id], 4)

        cv2.imshow(file_index, frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()

video_file1 = "rtsp-dir-1" 
video_file2 = "rtsp-dir-2"
tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, "1"), daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, "2"), daemon=True)
tracker_thread1.start()
tracker_thread2.start()
tracker_thread1.join()
tracker_thread2.join()
cv2.destroyAllWindows()
