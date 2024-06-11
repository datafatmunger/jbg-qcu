# Qcu simulator

Said "Q coo", like "cuckoo" (will create a cute bird logo later), is a POC, non-production, quantum universal gate simulator designed to run on Nvidia hardward via CUDA.

## Build

       nvcc -Isrc src/main.cu src/qsim.cu -o main

## Run

    .\main