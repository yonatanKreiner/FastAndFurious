from dateutil import parser
import json
import os
import cv2
import ntpath
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def split_video(video_full_name, output_path):
    config = __load_config_file()
    path, name = ntpath.split(video_full_name)
    date = parser.parse(config['timestamps'][name])
    fps = __get_fps(video_full_name)
    video_time_frame = int(config['video_time_frame'])
    __crop(video_full_name, output_path, video_time_frame)

def __crop(video_full_name, output_path, video_time_frame):
    clip = VideoFileClip(video_full_name)
    end_time = int(clip.duration / video_time_frame) * video_time_frame

    for start_time in xrange(0, end_time, video_time_frame):
        __crop_by_time(video_full_name, output_path, start_time, start_time + video_time_frame)
    
def __crop_by_time(video_full_name, output_path, start_time, end_time):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    output_path = os.path.join(output_path, str(start_time) + '-' + str(end_time) + '.avi')
    ffmpeg_extract_subclip(video_full_name, start_time, end_time, targetname=output_path)

def __get_fps(video_full_name):
    capture = cv2.VideoCapture(video_full_name)
    return capture.get(cv2.CAP_PROP_FPS)

def __load_config_file():
    with open('videos_config.json') as config_file:   
        return json.load(config_file, encoding='utf-8')