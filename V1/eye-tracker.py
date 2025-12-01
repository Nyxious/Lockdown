import cv2
video = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
print("Video Active:", video.isOpened())
eye_cascade = cv2.CascadeClassifier("eye_data.xml")
while True:
    ret, frame = video.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    detection = eye_cascade.detectMultiScale(frame, minSize=(100, 100))
    for (x, y, w, h) in detection:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    if (len(detection) > 0):
        cv2.imwrite("feed.jpg", frame)
    if not ret:
        break
