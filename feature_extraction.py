import os
import datetime
import csv
import optical_flow
from dateutil import parser
from features_extractors import *

def create_csv(videos_path, output_file, timestamps):
    out_file = open(output_file, "wb")
    writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writerow(['video_id', 'section_id', 'timestamp', 'movement'])

    for directory in os.listdir(videos_path):
        dir_path = os.path.join(videos_path, directory)

        out_file = open(output_file, "ab")
        writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

        if os.path.isdir(dir_path):
            for video in os.listdir(dir_path):
                video_id, offset, section_id = video.split('_')
                section_id = os.path.splitext(section_id)[0]

                row = [video_id, section_id]

                date = parser.parse(timestamps[video_id])
                date += datetime.timedelta(seconds=int(offset))
                row.append(str(date))

                features = __extract(os.path.join(dir_path, video))
                row.extend(features)

                writer.writerow(row)
        out_file.close()

def __extract(video_full_path):
    features_list = []

    motion = MotionDetectionExtractor()

    features_list.extend(motion.extract(video_full_path))

    return features_list
 
