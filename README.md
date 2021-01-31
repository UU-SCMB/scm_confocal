# scm_confocal
Library for importing and dealing with data from the Utrecht University Soft Condensed Matter groups confocal microscopes, including some general utility functions.

**[The documentation can be found here](https://maartenbransen.github.io/scm_confocal/)**

## Info
- created by:     Maarten Bransen
- email:          m.bransen@uu.nl

## Usage
The classes in this package typically require specific exporting formats from the confocal, and differ in specific implementation details because of this. I have attempted to be consistent in naming functions but each class has its own peculiarities based on personal need.

#### [SP8](https://maartenbransen.github.io/scm_confocal/sp8.html)
Data from the SP8 can be exported as tiff using the Leica LAS software (the microscope operation software), with the check marks for use RAW data checked. In principle data exported in color (so with a LUT applied) is accepted but not ideal and will return a warning for this reason. Now also supports the `.lif` files directly via the [readlif](https://github.com/nimne/readlif) library.

#### [Visitech Infinity](https://maartenbransen.github.io/scm_confocal/visitech.html)
Two classes are available, one for normal multy-dimensional acquisitions using MicroManager and one for our custom `faststack` driver.

#### [Utility functions](https://maartenbransen.github.io/scm_confocal/#scm_confocal.util)
Additionally, some utility functions for stacks on multidimensional microscopy data are included such as binning, rescaling, etc.
