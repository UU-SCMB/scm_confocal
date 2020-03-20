# scm_confocal
Library for importing and dealing with data from the Utrecht University Soft Condensed Matter groups confocal microscopes, including some general utility functions. **[The documentation can be found here](https://maartenbransen.github.io/scm_confocal/)**

## Info
- created by:     Maarten Bransen
- email:          m.bransen@uu.nl

## Installation instructions
Download the `scm_confocal` folder and place it in your `site-packages` location of your Anaconda installation. If you are unsure where this is located you can find the path of any already installed package, e.g. using numpy:
```
import numpy
print(numpy.__file__)
```
which may print something like
```
'<current user>\\AppData\\Local\\Continuum\\anaconda3\\lib\\site-packages\\numpy\\__init__.py'
```

Alternatively, you can install the master branch directly from github to the site packages folder using anaconda prompt and the git package.

## Usage
The classes in this package typically require specific exporting formats from the confocal, and differ in specific implementation details because of this. I have attempted to be consistent in naming functions but each class has its own peculiarities based on personal need.

#### SP8
Data from the SP8 *must* be exported as tiff using the Leica LAS software (the microscope operation software), with the check marks for use RAW data checked. In principle data exported in color (so with a LUT applied) is accepted but not ideal and will return a warning for this reason.

#### Visitech Infinity
Two classes are available, one for normal multy-dimensional acquisitions using MicroManager and one for our custom `faststack` driver.

#### Utility functions
Additionally, some utility functions for stacks on multidimensional microscopy data are included such as binning, rescaling, etc.
