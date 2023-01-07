from setuptools import setup

setup(name='MammogramPreprocessor',
      version='0.1',
      description='A custom class for preprocessing mammogram scans',
      url='https://github.com/Paul-Bacher/RSNA-Screening-Mammography-Breast-Cancer-Detection/MammogramPreprocessor',
      author='Paul Bacher',
      author_email='paulbacher881@gmail.com',
      license='CC',
      packages=['MammogramPreprocessor'],
      install_requires=[
            'gdcm',
            'time',
            'random',
            'numpy',
            'pandas',
            'matplotlib',
            'os',
            'glob',
            'tqdm',
            'pydicom',
            'cv2',
            'joblib'],
      zip_safe=False)