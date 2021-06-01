# cuda-dbscan
## Installation
First, make sure that git submodule dependencies are added:
`git submodule init`
`git submodule update`

Then, build the cudpp dependency using
`make cudpp`

Finally, build this project using
`make`

On systems where CUDA is not available, simply run
`make cpu`

## CPU Demo
Build with `make cpu`

Run with `./cpu_dbscan`

The demo will run the naive version and optimized version of the 
DBSCAN algorithm on 10,000 taxi points (see below for more information 
about the dataset -- I have included a portion of the data in the zip file).
It will print the runtime of each. Note that the optimized version of the 
algorithm is what I will be adapting for the GPU. The naive version
is used for checking the correctness of other implementations. My 
correctness checker is available in scripts/correctness.py. There is also
a visualizer in scripts/map_plot.py. In a future release, these will have 
make targets to run them with the necessary arguments.

## Dataset
This project uses New York City Yellow Taxi records for testing.
The data is publicly available at https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Note that latitude and longitude coordinates of pickups and dropoffs
are no longer recorded starting in 2016, so 2015 data was used.
