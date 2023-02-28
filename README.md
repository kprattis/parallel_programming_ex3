# Parallel computation of FGLT with CUDA

CUDA extension of the fglt library for the computation of the first five frequencies, 3rd exercise for the course Parallel and Distributed Systems, AUTH, 2023.

A lot of code has been taken from the github repository of the [fglt library](https://github.com/fcdimitr/fglt).

Tested with CUDA 12 and Ubuntu 22.04 OS.

Before compiling make sure CUDA is installed as well as the nvcc compiler.

## Compile:
* First change the makefile so that CC shows to the nvcc compiler path and CXX to the cpp compiler path.

* run 
```
 make 
``` 
from the project folder to compile the GPU version.

* run 
```
 make SEQ 
```
to compile the sequential version.

* there is an option ` make DEBUG ` to include some extra prints

## Run:
```
 ./fglt_cuda.out [MTXFILE]
```
or

```   
 ./fglt_seq.out [MTXFILE]
```

## Test:
``` 
make test 
```

Notice: Since the test is a small graph, GPU might appera slower than CPU execution. Please test with bigger graphs to sonfirm the speedup.