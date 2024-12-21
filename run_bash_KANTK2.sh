#!/bin/sh

#BSUB -q c02516

#BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -J MNISTKANTK2_20GB_12H

#BSUB -n 4

#BSUB -R "span[hosts=1]"

#BSUB -R "rusage[mem=20GB]"

#BSUB -W 12:00

#BSUB -o KANTK_MNIST2%J.out
#BSUB -e KANTK_MNIST2%e.err

python KANTK_MNIST2.py
