from ultralytics import YOLO
import pigpio
import math
kSG90_ID = 22
kImageWidth, kImageHeight = 640, 480
kFocalLengthMm = 32.0
kHFOVRadians = math.radians(102.0)
kFocalLengthToPixel = (kImageWidth / 2.0) / math.tan(kHFOVRadians / 2.0)
kReductionRate = 0.35
pigpioPi = pigpio.pi()
def getPWMConversion(rads):
    conversion = (2000.0 / math.pi) * rads + 500.0
    return conversion
modelV3 = YOLO("/home/lockdown/yolov8n_openvino_model/", task="detect")
lastAngularPosition = math.pi / 2.0
pigpioPi.set_servo_pulsewidth(kSG90_ID, 1500)
for frameData in modelV3.predict(source=0, stream=True, conf=0.75, show=False, classes=[0]):
    bbox = frameData.boxes
    if (len(bbox.cls) >= 1):
        bCenterWidth, bCenterHeight = bbox.xywh[0][0].item(), bbox.xywh[0][1].item()
        xDisplacementPixels = (bCenterWidth - kImageWidth / 2.0)
        thetaRadians = -math.atan(xDisplacementPixels / kFocalLengthToPixel) * kReductionRate
        newPosition = lastAngularPosition + thetaRadians
        if (newPosition > math.pi):
            lastAngularPosition = math.pi
        elif (newPosition < 0.0):
            lastAngularPosition = 0.0
        else:
            lastAngularPosition = newPosition
        pigpioPi.set_servo_pulsewidth(kSG90_ID, getPWMConversion(lastAngularPosition))
        print(f"Pixel Displacement: {xDisplacementPixels}", f"\nCenter X-Axis Coordinate: {bCenterWidth}", f"\nX-Angle Degrees: {math.degrees(thetaRadians)}")
