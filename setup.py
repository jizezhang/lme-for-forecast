from setuptools import setup

setup(name='lme',
      version='0.0',
      author='Jize Zhang',
      author_email='jizez@uw.edu',
      url='https://github.com/jizezhang/lme-for-forecast',
      packages=['lme'],
      package_dir={'lme': 'src/lme'},
      install_requires=['limetr', 'numpy'])
