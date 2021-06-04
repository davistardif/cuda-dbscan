# cuda-dbscan
## CS179 Project Turn-In Information
### Installation
The project is already built in this zip file. There are two executable: cpu-dbscan and gpu-dbscan
If for some reason you need to rebuild it, there are instructions below.
In short, `make cpu && make gpu` should do the trick.

### Usage
Each executable takes the same command line arguments:
`-n` : the number of points to supply to the algorithm
`-m` : the min points parameter supplied to the algorithm
`-e` : the epsilon parameter supplied to the algorithm
`-p` : toggle printing the clustering output (if omitted, timing information is printed)

There are also make targets for testing the project.
`make time-trial` will run the GPU and CPU versions on differing
input sizes

`make correctness` runs the CPU and GPU versions and compares their
output using a Python script. If there are no errors printed, the
outputs are in agreement. The CPU algorithm has been tested for correctness
in comparison to a naive implementation of the algorithm, so it should be correct.

### Project Description
This project is an implementation of the DBSCAN algorithm. It is a point
clustering algorithm that takes a set of x,y coordinates and finds clusters
of nearby points. The implementation is based on a paper by Wang, Gu, Shun (2019)
that provides an O(n log n) algorithm for DBSCAN (whereas it is naively O(n^2)).

### Results
The result of the programs is a clustering (which can be printed to stdout using the -p
option). The format of the files is a one line per input point with a
single character followed by an integer. Note that the line in the output file corresponds
to the index of the point in the input. The character represents the classification of the
input point: c for core points, b for border points and n for noise. The integer is a cluster
id (or -1) for noise points. All points with the same cluster ID are part of the same cluster.

### Performance Analysis
Unfortunately the GPU implementation is significantly slower than the CPU version.
I was unable to implement some steps of the algorithm on the GPU because of issues
incorporating 3rd party libraries (see below: Issues). Thus some steps are executed on
the GPU, then intermediate results are copied back to the host, more work is done,
and then copied to the GPU again, before finally being copied back to the CPU.
This adds a lot of overhead which could explain why the GPU implementation is so slow.
Also, it seems that this algorithm may not be particularly well suited to parallelization
with GPUs, since it relies heavily on hash tables. I've found the support for hash tables
on CUDA to be quite lacking. See below for more info. 

### Issues
The main issue preventing me from implementing the entire algorithm on the GPU is
the need for hash tables. I intended to use the CUDPP library for CUDA hash tables,
but ran into many issues. First, the API didn't have enough functionality exposed, so
I had to monkey-patch the library to expose it myself. Then, once I had the hash table working,
I added the gDel2D library for computing the Delaunay triangulation in CUDA. However, this
seems to have broken the CUDPP library. When compiled with gDel2D, I can no longer insert points
into the hash table without getting an obscure error from thrust / CUB radix_sort. My
hunch is that since both CUDPP and gDel2D rely on the thrust library, they may
need two different incompatible versions of it. In any case, after many hours of
debugging the libraries, I've given up on it and implemented the hash table steps
as CPU code. I also looked into other libraries for CUDA based hash tables such as
NVIDIAs new library "cuCollections", but it does not support multi-valued hash
tables at the moment. 


## Installation
First, make sure that git submodule dependencies are added:
`git submodule init`
`git submodule update`

Then, build the project dependencies using
`make cudpp`

Finally, build this project using
`make`

On systems where CUDA is not available, ignore the above and simply run
`make cpu`

## CPU Demo
Build with `make cpu`

Run with `./cpu_dbscan`

The demo will run the optimized version of the 
DBSCAN algorithm on 10,000 taxi points (see below for more information 
about the dataset -- I have included a portion of the data in the zip file).
It will then print the runtime. Note that the optimized version of the 
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
