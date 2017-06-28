import os
import video_splitter

video_path = 'D:/data/hackathon/'
video_name = '1.avi'
video_full_name = os.path.join(video_path, video_name)
output_path = os.path.join(video_path,os.path.splitext(video_name)[0])

video_splitter.split_video(video_full_name, output_path)