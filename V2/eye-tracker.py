import cv2

modelWeights = "/home/pi/Downloads/modelFiles/frozen_inference_graph.pb"
modelConfig = "/home/pi/Downloads/modelFiles/mobileNetV3.pbtxt"
modelNames = "/home/pi/Downloads/modelFiles/coco.names"
video = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
nameArray = []
with open(modelNames, "rt") as f:
    nameArray = f.read().rstrip("\n").split("\n")
model = cv2.dnn_DetectionModel(modelWeights, modelConfig)
model.setInputParams(1.0 / 127.5, (320, 320), (127.5, 127.5, 127.5), True)
while True:
    success, frame = video.read()
    classIds, confidence, box = model.detect(frame, confThreshold=0.6, nmsThreshold=0.7)
    if (len(classIds) > 0):
        for (objectId, c) in zip(classIds.flatten(), box):
            cv2.putText(frame, nameArray[objectId - 1],((c[0] + c[2])//2, (c[1] + c[3])//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0 ,0), 5, cv2.LINE_AA, False)
            cv2.rectangle(frame, (c[0], c[1]), (c[0] + c[2], c[1] + c[3]), (255, 0 ,0), 5)
            cv2.imwrite("feed.jpg", frame)
            break
    if not success:
        break
    

