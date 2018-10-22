# NE 204 Lab 2 ReadMe

This repo contains the necessary parts to perform the analysis and compile
the write-up of Lab 2.

Please use Python 2.7.
The following packages are required:
numpy, math, matplotlib, tables, csv

Two data files are available for downloading. These are csv files containing for each event the ADC value (determined by the SIS3302 DAQ on-board trapezoidal filter) and the rise time (t90-t10) for that pulse calculated with the calculate_rise_time function in the lab2_analysis_functions.py script. The rise time calculation in not run here to avoid the need for large amounts of data.

Running the analysis performs an evaluation of risetime selection cuts on the peak-to-compton ratio and peak-to-total ratio. Additionally, the sigggen.py script is run, which generates predicted signal shapes.

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
