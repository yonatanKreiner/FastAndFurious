import os
import datetime
import csv
import optical_flow
from dateutil import parser
from features_extractors import *

def create_csv(videos_path, output_file, timestamps):
    out_file = open(output_file, "wb")
    writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    features = ['video_id', 'section_id', 'timestamp', 'movement',"movement_avarage","movement_var", \
    "movement_std","num_of_elements_avarage","num_of_elements_var","num_of_elements_std", \
    "num_of_contours_avarage","num_of_contours_var","num_of_contours_std","max_contour_size", \
    "bg_change_above_15000","bg_change_above_45000","bg_change_above_75000","bg_change_above_105000", \
    "bg_change_above_135000","max_bg_diff","max_bg_diff_avarage","max_bg_diff_var","max_bg_diff_std"]
    writer.writerow(features)
    out_file.close()

    for directory in os.listdir(videos_path):
        dir_path = os.path.join(videos_path, directory)

        if os.path.isdir(dir_path):
            for video in os.listdir(dir_path):
                video_id, offset, section_id = video.split('_')
                section_id = os.path.splitext(section_id)[0]

                row = [video_id, section_id]

                date = parser.parse(timestamps[video_id])
                date += datetime.timedelta(seconds=int(offset))
                row.append(date.hour)

                features = __extract(os.path.join(dir_path, video))
                row.extend(features)

                out_file = open(output_file, "ab")
                writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

                writer.writerow(row)
                out_file.close()

def __extract(video_full_path):
    features_list = []

    motion = MotionDetectionExtractor()
    tom = TOM_FEATURES()

    features_list.extend(motion.extract(video_full_path))
    features_list.extend(tom.extract(video_full_path))

    return features_list
 
