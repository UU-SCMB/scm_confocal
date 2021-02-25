# scm_confocal
Library for importing and dealing with data from the Utrecht University Soft Condensed Matter groups confocal microscopes, including some general utility functions.

**[The documentation can be found here](https://maartenbransen.github.io/scm_confocal/)**

## Info
- created by:     Maarten Bransen
- email:          m.bransen@uu.nl

## Installation
This package can be installed directly from GitHub using pip:
```
pip install git+https://github.com/MaartenBransen/scm_confocal
```
### Anaconda
When using the Anaconda distribution, it is safer to run the conda version of pip as follows:
```
conda install pip
conda install git
pip install git+https://github.com/MaartenBransen/scm_confocal
```

## Usage
The classes in this package typically require specific exporting formats from the confocal, and differ in specific implementation details because of this. I have attempted to be consistent in naming functions but each class has its own peculiarities based on personal need.

#### SP8
Data from the SP8 can be saved as the native `.lif` files and imported using the [sp8_lif](https://maartenbransen.github.io/scm_confocal/#scm_confocal.sp8_lif) class, which is essentially a wrapper around the [readlif](https://github.com/nimne/readlif) library. This supports most (but not all) functions of the sp8, most notably lacking support for dirext xz imaging.

Alternatively, data can be exported using the Leica LAS software (the microscope operation software), with the check marks for use RAW data checked, and these can be loaded using [sp8_series](https://maartenbransen.github.io/scm_confocal/#scm_confocal.sp8_series) with support for all functions of the sp8. In principle data exported in color (so with a LUT applied) is accepted but not ideal and will return a warning for this reason.

#### Visitech Infinity
Two classes are available:

* [visitech_series](https://maartenbransen.github.io/scm_confocal/#scm_confocal.visitech_series) for normal multy-dimensional acquisitions using MicroManager
* [visitech_faststack](https://maartenbransen.github.io/scm_confocal/#scm_confocal.visitech_faststack) for xyzt data recorded using our custom `faststack` driver, which bypasses the z-stage feedback loop and metadata in favor of acquisition speed.

#### Utility functions
Additionally, some [utility functions](https://maartenbransen.github.io/scm_confocal/#scm_confocal.util) for stacks on multidimensional microscopy data are included such as binning, rescaling, etc.
