import os
import video_splitter
import split_to_sections
import feature_extraction

def __load_config_file(file_path):
    with open(file_path) as config_file:   
        return json.load(config_file, encoding='utf-8')

config_file = __load_config_file('videos_config.json')

videos_path = config_file[videos_path]
splitter = SectionsSplitter()

for video in os.listdir(videos_path):
    splitter.split(os.path.join(videos_path, video))

feature_extraction.create_csv(videos_path, config_file[output_csv], config_file[timestamps])

video_full_name = os.path.join(video_path, video_name)
output_path = os.path.join(video_path,os.path.splitext(video_name)[0])

