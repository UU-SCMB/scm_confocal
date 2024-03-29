from setuptools import setup, find_packages

setup(
    name="scm_confocal",
    version="1.7.5",
    author="Maarten Bransen",
    author_email="m.bransen@uu.nl",
    license='GNU General Public License v3.0',
    long_description=open('README.md').read(),
    packages=find_packages(include=["scm_confocal", "scm_confocal.*"]),
    install_requires=[
        "numpy>=1.19.2",
        "scipy>=1.6.0",
        "pillow>=9.2.0",
        "pims>=0.5",
        "readlif>=0.6.5",
        "numba>=0.50.0",
    ],
)
