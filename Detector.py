import os
from model.keras_yolo3.yolo import YOLO
import time
import cv2
import datetime
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import pandas as pd
import numpy as np


if __name__ == "__main__":
    # define a YOLO class instance
    yolo = YOLO(
        **{
            "model_path": 'model/trained_weights_final.h5',
            "anchors_path": 'model/keras_yolo3/model_data/yolo_anchors.txt',
            "classes_path": 'model/data_classes.txt',
            "score": 0.3,
            "gpu_num": 1,
            "model_image_size": (320, 320),
        }
    )

    # Make a dataframe for the prediction outputs
    out_df = pd.DataFrame(
        columns=[
            "image",
            "image_path",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "label",
            "confidence",
            "x_size",
            "y_size",
        ]
    )


    # Read in video from camera
    # For demostration, a demo video is in the folder
    vid = cv2.VideoCapture(os.path.join('wasp.mp4'))
    video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")  # int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (
        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    # print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
    out = cv2.VideoWriter(os.path.join('demoOut.mp4'),video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()


    while vid.isOpened():
        return_value, frame = vid.read()
        if not return_value:
            break
        # opencv images are BGR, translate to RGB
        frame = frame[:, :, ::-1]
        image = Image.fromarray(frame)
        out_pred, image = yolo.detect_image(image, show_stats=False)
        result = np.asarray(image)
        if len(out_pred)!=0:
            alert = [row[4] for row in out_pred]
            confidence = [row[5] for row in out_pred]
            print(alert,confidence)

     

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1

        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
            print(fps)

        timestamp = datetime.datetime.now()
        cv2.putText(
            result,
            timestamp.strftime("%A %d %B %Y %I:%M:%S%p"),
            (50, frame.shape[0] - 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 0, 255), 3
            )
	# cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", result)
        
        out.write(result[:, :, ::-1])
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
    vid.release()
    out.release()
    yolo.close_session()

