import os
import cv2
import time
from model_functions import *

### INPUT AND OUTPUT PATHS###
input_video = "./input_videos/3.mp4"
output_folder = "./output_videos"
output_video = os.path.join(output_folder, str(time.time()) + ".mp4")

### DL MODEL ###
model = make_model(input_shape=(32, 30), n_classes=6)
model.load_weights("model/weights.h5")
LABELS = ["JUMPING", "JUMPING_JACKS", "BOXING", "WAVING_2HANDS", "WAVING_1HAND", "CLAPPING_HANDS"]

### OPENCV INPUT AND OUTPUT CONF ###
W, H = 852, 480  # Width and height of the video
cap = cv2.VideoCapture(input_video)  # Input video object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Video write format
cap_rec = cv2.VideoWriter(output_video, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (W, H))  # Recording object


### INFERENCE LOOP ###
seq = np.zeros((32, 30), dtype=np.float32).tolist()  # Initial sample: We will append new key points to this list
while True:
    success, img_or = cap.read()
    if not success:
        break

    try:

        img = cv2.resize(img_or.copy(), (W, H), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR2RGB because the pose_model works in RGB format
        results = pose_model.process(img)  # result is the pose object
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        p, img = get_processed_pose(img, results, True, W, H)  # This will return annotated image and pose as a array
        seq.append(p)  # append new pose
        seq.pop(0)  # remove oldest pose so that our buffer will be same length
        pred = np.round(
            model.predict(np.array([np.array(seq, dtype=np.float32)])))  # Final prediction in one hot format
        idx = np.where(pred == 1.0)[1][0]  # Final prediction index
        if sum(seq[10]) == 0:
            act = ""
        else:
            act = LABELS[idx]  # Final prediction

    except:
        pass
    img_plotted = cv2.putText(img=img.copy(), text=act, org=(40, 80), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2,
                              color=(0, 0, 255), thickness=3)
    cap_rec.write(img_plotted)
    cv2.imshow("out", img_plotted)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cap_rec.release()
