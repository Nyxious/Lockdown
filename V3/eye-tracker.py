from ultralytics import YOLO
import pigpio
import math
kSG90x_ID, kSG90y_ID = 22, 17
kImageWidth, kImageHeight = 640.0, 480.0
kFocalLengthMm = 32.0
kHFOVRadians, kVFOVRadians = math.radians(102.0), math.radians(57.0)
kFocalLengthToPixelx = (kImageWidth / 2.0) / math.tan(kHFOVRadians / 2.0)
kFocalLengthToPixely = (kImageHeight / 2.0) / math.tan(kVFOVRadians / 2.0)
kXReductionRate, kYReductionRate = 0.35, 0.40
pigpioPi = pigpio.pi()
def getPWMConversion(rads):
    conversion = (2000.0 / math.pi) * rads + 500.0
    return conversion
modelV3 = YOLO("/home/lockdown/yolov8n_openvino_model/", task="detect")
lastXAngularPosition = math.pi / 2.0
hardLimitY = 99.0 * (math.pi / 180.0)
lastYAngularPosition = hardLimitY
pigpioPi.set_servo_pulsewidth(kSG90x_ID, 1500)
pigpioPi.set_servo_pulsewidth(kSG90y_ID, 1600)
for frameData in modelV3.predict(source=0, stream=True, conf=0.75, show=False, classes=[0]):
    bbox = frameData.boxes
    if (len(bbox.cls) >= 1):
        bCenterWidth, bCenterHeight = bbox.xywh[0][0].item(), bbox.xywh[0][1].item() #bbox.xyxy[0][1] for near head tracking
        xDisplacementPixels = (bCenterWidth - kImageWidth / 2.0)
        yDisplacementPixels = (bCenterHeight - kImageHeight / 2.0)
        thetaXRadians = -math.atan(xDisplacementPixels / kFocalLengthToPixelx) * kXReductionRate
        thetaYRadians = -math.atan(yDisplacementPixels / kFocalLengthToPixely) * kYReductionRate
        newXPosition = lastXAngularPosition + thetaXRadians
        newYPosition = lastYAngularPosition + thetaYRadians
        if (newXPosition > math.pi):
            lastXAngularPosition = math.pi
        elif (newXPosition < 0.0):
            lastXAngularPosition = 0.0
        else:
            lastXAngularPosition = newXPosition
        pigpioPi.set_servo_pulsewidth(kSG90x_ID, getPWMConversion(lastXAngularPosition))
        if (newYPosition > math.pi):
            lastYAngularPosition = math.pi
        elif (newYPosition < hardLimitY):
            lastYAngularPosition = hardLimitY
        else:
            lastYAngularPosition = newYPosition
        pigpioPi.set_servo_pulsewidth(kSG90y_ID, getPWMConversion(lastYAngularPosition))
        #print(f"Pixel Displacement: {xDisplacementPixels}", f"\nCenter X-Axis Coordinate: {bCenterWidth}", f"\nX-Angle Degrees: {math.degrees(thetaRadians)}")
