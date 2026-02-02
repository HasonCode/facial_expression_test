import cv2 as cv
import numpy as np
import time 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
timer, past_time = 0, 0

cam = cv.VideoCapture(0)

model_path = "./blaze_face_short_range.tflite"


if not cam.isOpened():
    print("camera not opening")
    exit()

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode
latest_res = {"result":None}
def print_result (result, output_image, timestamp_ms ):
    # print(f'face detector result {result}')
    latest_res["result"]=result
    pass


counter = 0
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    min_detection_confidence=0.7,
)
timed_shots = True
prev_timer = timer
result = None
cropped = None
cropped_init = False

def push_next_stamp(prev_time:int, initial_time:int) -> int:  
    cur = (time.monotonic_ns()-initial_time)//1000000
    if cur <= prev_time:
        cur+=1
    return int(cur)

initial_time = time.monotonic_ns()
prev_timer  = 0


def save_label(label:str, label_directory:str, number:int, frame):
    cv.imwrite(f"{label_directory}/{label}/{label}_{number}.png", frame)


neutral_count,smile_count,blink_count,frown_count,anger_count=923,1,1344,1,1239
with FaceDetector.create_from_options(options) as detector:
    try:
        while True:
            timer=push_next_stamp(prev_timer, initial_time)
            # print(f"timer_val: {timer}")
            happened, frame = cam.read()
            if (not happened):
                print("huh")
                exit()
            
            gray_pic = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            if timer!=prev_timer:result= detector.detect_async(mp_image,int(timer))
            h,w,_ = frame.shape
            if latest_res["result"]:
                # print(result.Detection)
                for detect in latest_res["result"].detections:
                    bbox = detect.bounding_box
                    x,y,w,h = bbox.origin_x, bbox.origin_y, bbox.width,bbox.height
                    cropped = frame[x-int(w*0.5):x+int(w*1.5), y-int(h*.5):y+int(h*1.5)]
                    # cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    cropped_init = True
            if cropped_init:
                cv.imshow("picture",frame)
            else:
                cv.imshow("picture",frame)
            key_collect = cv.waitKey(0)
            if (key_collect == ord('q')):
                break
            elif (key_collect==ord("1")): #neutral
                save_label("neutral", "dataset", neutral_count,cv.resize(cropped,(256,256)))
                neutral_count+=1
            elif (key_collect==ord("2")): #smile
                save_label("smile", "dataset", smile_count,cv.resize(cropped,(256,256)))
                smile_count+=1
            elif (key_collect==ord("3")): #blink
                save_label("blink", "dataset", blink_count,cv.resize(cropped,(256,256)))
                blink_count+=1
            elif (key_collect==ord("4")): #frown
                save_label("frown", "dataset", frown_count,cv.resize(cropped,(256,256)))
                frown_count+=1
            elif(key_collect==ord("5")): #anger
                save_label("anger", "dataset", anger_count,cv.resize(cropped,(256,256)))
                anger_count+=1

            prev_timer = timer
    except:
        print(timer,prev_timer)
    cam.release()
    cv.destroyAllWindows()