# Fractal-Aggregates-with-tunable-dimension
Program to create a hierachical fractal cluster to mimic the arrangement of synaptic vesicles.  


## RECOMMENDAION TO RUN THE SCRIPTS

The scripts were executed with Pycharm and in cell mode. The cell delimiters are marked by two consecutive number-signs ##.

Version of Python interpreter: 3.9

Required Python-libraries: Numpy, Scipy, Matplotlib, Pytorch, math, random, Numpy-Quaternion, copy

## DESCRIPTION 

The code presented here is based on the appraches of Tomchuk et al.
https://www.sciencedirect.com/science/article/abs/pii/S0927775720309249?via%3Dihub

The program is designed to create a fractal aggregate cluster consisting of 2^o particles. The fractal dimension of the cluster, which is related to the compactness, can be tuned with the simulation parameters D and k. The procedure for creating the clusters is hierarchical: clusters of the same size are combined to form larger clusters, where the algorithm does not require any kinetic calculations at all.
The procedure is as follows: first a cluster is created and its radius of gyration calculated, then a second cluster is created and the radius of gyration calculated.  From the two Rg, a distance 'gamma' can now be determined in which the two centres of mass of the clusters should lie. After the clusters are moved to this distance, they rotate until they touch but do not overlap. After successful connection, the two clusters form a new larger cluster. The radius of gyration of the new clsuter can be calculated from the Rg of the previous old clusters. This sequence can be repeated until the desired cluster size is reached. For details see also tomchuk et al. 

Afterwards the one-dimensional structure factor of the cluster is computed and plottet  


