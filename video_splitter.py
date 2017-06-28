import os
import utils

def split_video(video_id, video_full_name, output_path, video_time_frame):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    end_time = int(utils.get_duration(video_full_name) / video_time_frame) * video_time_frame

    for start_time in xrange(0, end_time, video_time_frame):
        __crop_by_time(video_id, video_full_name, output_path, str(start_time), video_time_frame)
    
def __crop_by_time(video_id, video_full_name, output_path, start_time, duration):
    output_path = os.path.join(output_path, video_id + '_' + start_time + '.avi')
    #ffmpeg_extract_subclip(video_full_name, start_time, duration, targetname=output_path)

    cmd = 'ffmpeg -i ' + video_full_name + ' -vcodec copy -acodec copy -copyinkf -ss ' \
    + start_time + ' -t ' + str(duration) + ' '\
             + output_path
           
    print(cmd)
    os.system(cmd)