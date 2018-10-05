# NE 204 Lab 1 ReadMe

This repo contains the necessary parts to perform the analysis and compile
the write-up of Lab 1. Please use Python 2.7.

The following packages are required:
numba, numpy, math, matplotlib, lmfit, tables, csv, time

Note that this does not perform the full analysis due to the inconvenience of
downloading large data files. The data includes tenevents.txt, which contains
digitized signals for 10 events, with a 10 ns sampling interval. The other text
files (3 files with the names 'co_energies.txt', 'cs_energies.txt', 'am_energies.txt') contain the amplitudes of signals from much larger data sets after the application of a trapezoidal filter that was optimized for energy resolution.

The analysis command does all of the parts of the analysis which can be demonstrated using these smaller data sets.

## How to run analysis

1. download data:
```
make data
```
2. run analysis:
```
make analysis
```
3. run software test suite:
```
make test
```
## How to compile report

4. generate report:
```
make
```
## How to clean
5. clean:
```
make clean
```
