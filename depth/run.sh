#!/bin/bash

if [ ! -d "losses" ]; then
  echo "Creating losses directory."
  mkdir losses
fi

if [ ! -d "plots" ]; then
  echo "Creating plots directory."
  mkdir plots
fi

for i in {0..1}
do
    python relu.py $i
    python gdln.py $i
done
