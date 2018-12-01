# NE 204 HPGe Final Project ReadMe

This repo contains the necessary parts to perform the analysis and compile
the write-up of my NE204 final project, based around Lab 5 and focusing on
determining interaction depth in an HPGe double-sided strip detector.

Please use Python 2.7.
The following packages are required:
numpy, math, matplotlib, tables, csv

Data files are available for downloading.... but are huge, so the calibration.py file,
though it can be run, needs 4 data files which can be provided (email jszornel@berkeley.edu).
Currently the analysis only performs timing.py, which relies on only 2 data files.
These are still pretty large (sorry :()

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
