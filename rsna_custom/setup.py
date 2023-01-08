from setuptools import setup, find_packages

setup(
    name='rsna_custom',
    version='0.1',
    packages=find_packages(),
    description='A custom package for the RSNA Breast Cancer Detection competition',
    author='Paul Bacher',
    author_email='paulbacher881@gmail.com',
    license='Free',
    install_requires=['python-gdcm', 'pydicom', 'pylibjpeg', 'numpy', 'pandas',
    'opencv-python', 'matplotlib', 'tqdm', 'joblib'],
)