#!/bin/bash

CHANNELS="-c c3i_test2/label/dpcpp -c intel -c defaults --override-channels"

#VERSIONS="--numpy 1.18"
VERSIONS="${VERSIONS} --python 3.8"

TEST="--no-test"

conda build \
  $TEST \
  $VERSIONS \
  $CHANNELS \
  conda-recipe
