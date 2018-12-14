# NE 204 HPGe Final Project ReadMe

This repo contains the necessary parts to perform the analysis and compile
the write-up of my NE204 final project, based around Lab 5 and focusing on
determining interaction depth in an HPGe double-sided strip detector.

The depth of interaction will be extracted for single energy deposition events
in a few different ways. Running the analysis included here will use a linear
fit to determine the interaction depths. However, the library method- whereby
signals are compared with a library of simulated signals- is excluded
(commented out) due only to the fact that the process is very slow (each raw
signal is compared to each library signal, which includes a time-alignment step
where the signal shifted in time to find the best fit. This is all quite slow).

One sample signal is evaluated, purely for demonstration.


Please use Python 2.7.
The following packages are required:
numpy, math, matplotlib, tables, csv, lmfit, numba, scipy, time

Data files are available for downloading.... but are quite large...

The calibration.py file, though it can be run, requires 4 data files which are not
included but can be provided (email jszornel@berkeley.edu).

Currently the analysis only performs timing.py, which relies on only 1 data file
which is still pretty large, sorry :(

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
