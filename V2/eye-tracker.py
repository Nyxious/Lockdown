import cv2
import pigpio
import math
modelWeights = "/home/pi/Downloads/modelFiles/frozen_inference_graph.pb"
modelConfig = "/home/pi/Downloads/modelFiles/mobileNetV3.pbtxt"
modelNames = "/home/pi/Downloads/modelFiles/coco.names"
nameArray = []
focalLen = 32
video = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
pi = pigpio.pi()
with open(modelNames, "rt") as f:
    nameArray = f.read().rstrip("\n").split("\n")
model = cv2.dnn_DetectionModel(modelWeights, modelConfig)
model.setInputParams(1.0 / 127.5, (320, 320), (127.5, 127.5, 127.5), True)
def getAngleConversion(rads):
    rads = rads + (math.pi / 2)
    conversion = (2000 / math.pi) * rads + 500
    return conversion
while True:
    success, frame = video.read()
    classIds, confidence, box = model.detect(frame, confThreshold=0.6, nmsThreshold=0.7)
    print(len(classIds)) # Testing to check if theres anything inside of the classIds array
    if (len(classIds) > 0):
    
        for (objectId, c) in zip(classIds.flatten(), box):
            
            print(nameArray[objectId - 1]) # Prints the object detected using the coco list
            if (nameArray[objectId - 1] == "person"):
                #new2
                coordinateX, widthCenter = (c[0] + (c[2]/2)), frame.shape[1] // 2
                print(coordinateX, widthCenter)
                angle = getAngleConversion(math.atan((coordinateX - widthCenter) / focalLen))
                print(angle)
                pi.set_servo_pulsewidth(22, angle)
                cv2.imwrite(nameArray[objectId - 1]+".jpg", frame)
                
                break
            
    if not success:
        break
