#!/bin/bash

module purge
module load anaconda/3/2023.03

# The following command replaces `conda init` for the current session
# without touching the .bashrc file:
eval "$(conda shell.bash hook)"