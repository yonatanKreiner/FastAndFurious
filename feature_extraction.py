import os
import csv
import features_extractors
import optical_flow
from dateutil import parser

def create_csv(videos_path, output_file, timestamps):
    out_file = open(output_file, "wb")
    writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
    writer.writerow('video_id,section_id,timestamp,movement')

    for video in os.listdir(videos_path):
        video_id, offset, section_id = video.split('_')
        section_id = os.path.splitext(section_id)[0]

        row[video_id, section_id]

        date = parser.parse(timestamps[video_id])
        date += datetime.timedelta(seconds=offset)
        row.append(date)

        features = __extract(os.path.join(videos_path, video))
        row.append(features)

        writer.writerow(video_id)
    
    out_file.close()

def __extract(video_full_path):
    features_list = []

    motion = MotionDetectionExtractor()
    frames = FramesCompareExtractor()

    features_list.append(motion.extract(video_full_path))
    features_list.append(frames.extract(video_full_path))
    features_list.append(optical_flow.dense_optical_flow(video_full_path))

    return features_list
 
