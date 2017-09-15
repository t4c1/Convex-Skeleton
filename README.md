# Convex-skeleton

This repository contains an implementation of algorithm for creating [convex skeletons of a network (graph)](). 
A convex skeleton of a network is its connected subgraph, which is as [convex](https://arxiv.org/pdf/1608.03402.pdf) as 
possible. Finding optimal skeleton for given number of links is hard, so the algorithm instead removes links from a 
network one at a time, maximizing network clustering coefficient.

# Building

This implementation depends on [Network convexity algorithm available here](https://github.com/t4c1/Graph-Convexity).
To enable parallelism by OpenMP appropriate compiler switch should be set. Example compiling comand line application 
using g++:

`g++ -std=c++11 -fopenmp -O3 algo.cpp convexSkeleton.cpp -o convex-skeleton`

# Command line usage

`convex-skeleton input [n_skeletons [step]]`

`input`: path to pajek (.net) file, containing the network. Do not include .net file extension in input - it will 
be added automaticvly. This is the only mandatory parameter.

`n_skeletons`: number of skeletons to create for each number of links. Skeletons will be very similar, but if two links
increase clustering by the same amount random one will be removed. default is 100.

`step` fraction of links to remove in each step. Regardles of `step` parameter always removes at least 1 link. Default is 0.01. 

Resulting skeletons will be saved in in a subfolder of where input file is. Subfolder will have same name as input file.
Output files will be named according to scheme: [number of removed links]_[iteration].net.