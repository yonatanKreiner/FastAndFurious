import os
import csv
import features_extractors

def create_csv(videos_path, output_file, timestamps):
    out_file = open(output_file, "wb")
    writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
    writer.writerow('video_id,section_id,timestamp,movement')

    for video in os.listdir(videos_path):
        video_id, offset, section_id = video.split('_')
        section_id = os.path.splitext(section_id)[0]

        row[video_id, section_id]
        row.append(timestamps[video_id] + offset)

        features = __extract(os.path.join(videos_path, video))
        row.append(features)

        writer.writerow(video_id)
    
    out_file.close()

def __extract(video_full_path):
    features_list = []

    motion = MotionDetectionExtractor()
    motion = MotionDetectionExtractor()
    motion = MotionDetectionExtractor()
    motion = MotionDetectionExtractor()

    features_list.append(motion.extract(video_full_path))
    features_list.append(motion.extract(video_full_path))
    features_list.append(motion.extract(video_full_path))
    features_list.append(motion.extract(video_full_path))

    return features_list
 
