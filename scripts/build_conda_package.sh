#!/bin/bash

PYTHON_VERSION=$1

VERSIONS="--python $PYTHON_VERSION"
TEST="--no-test"

conda build \
  $TEST \
  "$VERSIONS" \
  "$CHANNELS" \
  conda-recipe
