
Instructions to run the program luh.jl

We have ran in julia 1.9.0 in a ubuntu linux system.

Place the luh.jl program in a folder along with the .txt data files
From within this folder open a terminal and run julia:

$julia

Inside julia call

julia> include("luh.jl")

The program will ask for the data file to be used: give the name of
one of .txt files that is in the folder.
For example: luem03.txt, whose results were presented in Fig.(1) of the paper
and is the Light Up of Fig.(2).

The following packages must be installed
DelimitedFiles, OffsetArrays, Crayons, LinearAlgebra, Statistics, PyPlot, Printf


