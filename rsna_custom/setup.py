from setuptools import setup, find_packages

setup(
    name='rsna_custom',
    version='0.2',
    packages=find_packages(),
    description='A custom package for the RSNA Breast Cancer Detection competition',
    author='Paul Bacher',
    author_email='paulbacher881@gmail.com',
    license='Free',
    install_requires=['numpy', 'pandas', 'opencv-python', 'pydicom', 'tqdm',
    'joblib', 'matplotlib', 'python-gdcm', 'pylibjpeg', 'dicomsdl']
)

