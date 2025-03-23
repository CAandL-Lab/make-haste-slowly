#!/bin/bash

if [ ! -d "losses" ]; then
  echo "Creating losses directory."
  mkdir losses
fi

if [ ! -d "mds" ]; then
  echo "Creating mds directory."
  mkdir mds
fi

if [ ! -d "svs" ]; then
  echo "Creating svs directory."
  mkdir svs
fi

if [ ! -d "plots" ]; then
  echo "Creating plots directory."
  mkdir plots
fi

python relu.py
python gdln.py
python gdln_single.py
python race.py
python closed.py
python replot.py
