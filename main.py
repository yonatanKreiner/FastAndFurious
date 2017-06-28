import os
import video_splitter
import split_to_sections
import feature_extraction
import utils

config_file = utils.load_config_file('videos_config.json')

videos_path = config_file['videos_path']
output_path = config_file['output_path']
video_frame_rate = int(config_file['video_time_frame'])

splitter = split_to_sections.SectionsSplitter()

for video_id, video_name in enumerate(os.listdir(videos_path)):
    video_full_name = os.path.join(videos_path, video_name)

    if not os.path.isfile(video_full_name):
        continue
    
    video_id = str(video_id)
    output_full_path = os.path.join(output_path, video_id)
    video_splitter.split_video(video_id, video_full_name, output_full_path, video_frame_rate)
    import time
    for splitted_video_name in os.listdir(output_full_path):
        splitted_video_path = os.path.join(output_full_path, splitted_video_name)
        splitter.split(splitted_video_path)
        time.sleep(5)

feature_extraction.create_csv(videos_path, config_file['output_csv'], config_file['timestamps'])

