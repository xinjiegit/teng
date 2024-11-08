#!/bin/bash

# Restrict JAX to a single GPU (device 0)
export CUDA_VISIBLE_DEVICES=0



# Run the Python script (adjust the path to point to `teng.py`)
python teng.py