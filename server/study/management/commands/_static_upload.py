import os
import shutil


def upload_video(video_path, static_path, video_id, ext='.mp4'):
    assert os.path.exists(video_path)
    assert os.path.splitext(video_path)[1] == ext
    assert os.path.exists(static_path)
    assert os.path.isdir(static_path)

    dir_path = video_id + ext
    dest_path = os.path.join(static_path, dir_path)
    shutil.copy(video_path, dest_path)
    return dir_path
