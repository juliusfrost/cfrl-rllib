from setuptools import setup

setup(name="doodad",
      version='0.1',
      description='A job launching library for docker, EC2, etc.',
      url='https://github.com/justinjfu/doodad',
      author='Justin Fu, Vitchyr Pong',
      author_email='justinjfu@eecs.berkeley.edu',
      license='MIT',
      packages=['doodad'],
      install_requires=[
        'boto3',
        'boto',
        'cloudpickle',
        'awscli'
      ],
      zip_safe=False)