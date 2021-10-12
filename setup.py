from setuptools import setup, find_packages

setup(
    name="scm_confocal",
    version="1.4.3",
    author="Maarten Bransen",
    author_email="m.bransen@uu.nl",
    license='GNU General Public License v3.0',
    long_description=open('README.md').read(),
    packages=find_packages(include=["scm_confocal", "scm_confocal.*"]),
    install_requires=[
        "numpy>=1.19.2",
        "scipy>=1.6.0",
        "pillow>=6.2.1",
        "pims>=0.5",
        "opencv-python>=3.0.0",
        "readlif>=0.6.0",
        "numba>=0.50.0",
    ],
)
