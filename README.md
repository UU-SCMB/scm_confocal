# scm_confocal
Library for importing and dealing with data from the Utrecht University Soft Condensed Matter groups confocal microscopes

## info
- created by:     Maarten Bransen
- email:          m.bransen@uu.nl
- last updated:   06-11-2019

## installation
Download the `scm_confocal` folder and place it in your `site-packages` location of your Anaconda installation. If you are unsure where this is located you can find the path of any already installed package, e.g. using numpy:
```
import numpy
print(numpy.__file__)
```

## use
The classes in this package typically require specific exporting formats from the confocal, and differ in specific implementation details because of this. I have attempted to be consistent in naming functions but each class has its own peculiarities based on personal need.

### SP8
Data from the SP8 *must* be exported as tiff using the Leica LAS software (the microscope operation software), with the check marks for use RAW data checked. In principle data exported in color (so with a LUT applied) is accepted but not ideal and will return a warning for this reason.

### Visitech Infinity
Two classes are available, one for normal multy-dimensional acquisitions using MicroManager and one for our custom `faststack` driver.
