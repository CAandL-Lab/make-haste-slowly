#!/bin/bash

if [ ! -d "losses" ]; then
  echo "Creating losses directory."
  mkdir losses
fi

if [ ! -d "svs3" ]; then
  echo "Creating svs directory."
  mkdir svs3
fi

if [ ! -d "svs4" ]; then
  echo "Creating svs directory."
  mkdir svs4
fi

if [ ! -d "svs5" ]; then
  echo "Creating svs directory."
  mkdir svs5
fi

if [ ! -d "plots" ]; then
  echo "Creating plots directory."
  mkdir plots
fi

python closed3.py
python closed4.py
python closed5.py
python gdln3.py
python gdln4.py
python gdln5.py
python relu3.py
python relu4.py
python relu5.py
python replot.py
