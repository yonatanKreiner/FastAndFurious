import cv2
import json
from moviepy.video.io.VideoFileClip import VideoFileClip

def load_config_file(file_path):
    with open(file_path) as config_file:   
        return json.load(config_file, encoding='utf-8')

def get_fps(video_full_name):
    capture = cv2.VideoCapture(video_full_name)
    return capture.get(cv2.CAP_PROP_FPS)

def get_duration(video_full_name):
    clip = VideoFileClip(video_full_name)
    return clip.duration