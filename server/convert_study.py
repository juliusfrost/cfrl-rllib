import os
import shutil
import argparse
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-dir', required=True, type=str,
                        help='The path to directory containing the html study documents.')
    parser.add_argument('--template-dir', default=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'templates'))
    args = parser.parse_args()
    if os.path.exists(args.study_dir):
        for file in os.listdir(args.study_dir):
            path = os.path.join(args.study_dir, file)
            if os.path.isfile(path) and os.path.splitext(file)[1] == '.html':
                print(f'converting study located at {path}')
                study_id = uuid.uuid4()

                # dest_path = os.path.join(args.template_dir, 'study', file)
                # print(f'copying file {path} to {dest_path}')
                # shutil.copy(path, dest_path)
    else:
        print(f'The path {args.study_dir} does not exist.')


if __name__ == '__main__':
    main()
