import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', default='videos')
    args = parser.parse_args()
    if not os.path.exists(args.video_dir):
        raise FileNotFoundError(f'Video directory does not exist: {args.video_dir}')
    

if __name__ == '__main__':
    main()
