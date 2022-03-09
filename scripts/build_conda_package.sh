#!/bin/bash

PYTHON_VERSION=$1

VERSIONS="--python $PYTHON_VERSION"
TEST="--no-test"

# shellcheck disable=SC2086
conda build \
  $TEST \
  $VERSIONS \
  $CHANNELS \
  conda-recipe
