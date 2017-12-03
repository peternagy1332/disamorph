from distutils.core import setup

setup(
    name='Disamorph - Morphological Disambiguator',
    version='1.0',
    author='Peter Nagy',
    author_email='peter.nagyy1332+disamorph@gmail.com',
    url='https://github.com/peternagy1332/disamorph',
    packages=['disamorph'],
    license='MIT',
    install_requires=[
        'h5py==2.7.0',
        'numpy==1.13.3',
        'args==0.1.0',
        'tensorflow-gpu==1.4.0'
    ],
    extras_requires=[
        'floyd-cli==0.10.22'
    ]
)