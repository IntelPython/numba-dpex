#!/bin/bash

create_channel() {
  # cd $GITHUB_WORKSPACE
  ARTIFACT=/localdisk/work/spokhode/miniconda3/envs/build-env/conda-bld/linux-64/numba-dppy-0.14.4-py38h2bc3f7f_2.tar.bz2

  mkdir -p channel/linux-64
  cp $ARTIFACT channel/linux-64

  cd channel
  conda index

  GITHUB_WORKSPACE=/localdisk/work/spokhode
  CHANNELS="-c $GITHUB_WORKSPACE/channel -c c3i_test2/label/dpcpp -c intel -c defaults --override-channels"
}

CHANNELS="-c intel -c defaults --override-channels"

conda install numba-dppy pytest $CHANNELS --name base
