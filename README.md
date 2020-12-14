# Replication code for Errickson, Keller, Collins, Srikrishnan and Anthoff "Improving the Climate Model Calibration Reduces the Social Cost of Methane"

This repository holds all code required to replicate the results of:

Errickson, Keller, Collins, Srikrishnan and Anthoff (forthcoming) "Improving the Climate Model Calibration Reduces the Social Cost of Methane".

## Hardware and software requirements

You need to install [Julia](http://julialang.org/) and [R](https://www.r-project.org/) to run the replication code. We tested this code on Julia version 1.5.3 and R version 4.0.3.

Make sure to install both Julia and R in such a way that both the Julia and R binary are on the `PATH`.

Running the complete replication code on Windows on a system with an Intel Xeon W-2195 19 core CPU, 128 GB of RAM and a 1 TB SSD hard drive takes about four days.

## Running the replication script

To recreate all outputs and figures for this paper, open a OS shell and change into the folder where you downloaded the content of this replication repository. Then run the following command to compute all results:

```
julia --procs auto src/main.jl
```

Once that is finished, run the following command to create all figures:

```
rscript src/create_figures.r
```

Both scripts are configured such that they download and install any required Julia and R packages to run the replication code automatically.

## Result and figure files

All results and figures will be stored in the folder `results`.
