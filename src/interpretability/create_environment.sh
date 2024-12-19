#!/usr/bin/env bash

conda env create -f environment.yml

conda activate processing
python -m pip install --upgrade -r requirements_WJ.txt
