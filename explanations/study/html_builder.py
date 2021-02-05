import os

import imageio

from explanations.study.builder import StudyBuilder


def iframe_video(image_path, video_path):
    vid = imageio.get_reader(os.path.join(video_path, image_path), 'mp4')
    img = vid.get_data(0)
    h, w, _ = img.shape
    return f'<iframe height={h} width={w} src={image_path}></iframe>\n'


class HTMLStudyBuilder(StudyBuilder):

    def __init__(self, save_dir, root_dir, explanation_method, num_trials, config, name):
        super().__init__(save_dir, root_dir, explanation_method, num_trials, config, name)
        self.file_name = os.path.join(save_dir, f'{name}.html')
        self.html = ''

    def build_intro(self, title, intro_text):
        self.html += (
            f'<!DOCTYPE html> '
            f'<html>\n'
            f'    <head>\n'
            f'        <title>{title}</title>\n'
            f'    </head>\n'
            f'    <body>\n'
            f'<h1>{title}</h1>\n'
            f'<p>{intro_text[0]}</p>\n'
            f'<p>{intro_text[1]}</p>\n'
            f'<p>{intro_text[2]}</p>\n'
        )

    def build_outro(self):
        self.html += (
            '    </body>\n'
            '<html>\n'
        )

    def add_explanations(self, video_path, root_dir, heading_text, body_text):
        self.html += (
            f'<h3>{heading_text}</h3>\n'
            f'<p>{body_text}</p>\n'
        )
        try:
            self.html += iframe_video(video_path, root_dir)
        except FileNotFoundError:
            print(f'Could not find image {root_dir}')

    def add_evaluations(self, video_path, root_dir, heading_text, body_text, question_text):
        self.html += (
            f'<h3>{heading_text}</h3>\n'
            f'<p>{body_text}</p>\n'
        )
        try:
            self.html += iframe_video(video_path, root_dir)
        except FileNotFoundError:
            print(f'Could not find image {root_dir}')
        self.html += f'<p>{question_text}</p>'

    def save(self):
        with open(self.file_name, 'w') as f:
            f.write(self.html)

    def trial_heading(self, trial_heading_text):
        self.html += f'<h2>{trial_heading_text}</h2>\n'
