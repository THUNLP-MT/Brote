import json
from tqdm import tqdm
import cv2


TEXT_PATH = "the path for the text json of mmicl"
BASE_VIDEO_PATH = "the base path for the video" # like "../images/ivqa_video"


def process(file):

    json_objects = []
    
    with open(file, "r") as file:
        for line in file:
            json_object = json.loads(line)  # Convert line to JSON object
            json_objects.append(json_object)

    for line_json in tqdm(json_objects):

        video_path = line_json["input_image"]

        parts = video_path.split('/')
        parts[:3] = "../images/msrvtt_video".split('/') # for mmicl data format
        video_path = '/'.join(parts)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = total_frames // 8
        for i in range(8):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ret, frame = cap.read()
            if ret:
                path = video_path.replace(".mp4", f"_image{i}.jpg")
                path = path.replace(".webm", f"_image{i}.jpg")
                path = path.replace("_video", "")
                cv2.imwrite(path, frame)
            else:
                print(f"Error: Could not read frame {i}.")
        cap.release()

if __name__ == "__main__":
    process(TEXT_PATH)


    