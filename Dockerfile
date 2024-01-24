# syntax=docker/dockerfile:1.3
# NB: at least 1.3 is needed to benefit from ARG expansion in bind mount arguments
ARG PYTHON_VERSION=3.9.17

# Driver args
# print on gpu is broken for 22.43.24595.30 + igc-1.0.12812.26 refer to these
# versions for testing (all tests pass on them):
# ARG CR_TAG=22.43.24595.30
# ARG IGC_TAG=igc-1.0.12504.5
# ARG CR_TAG=latest
# ARG IGC_TAG=latest
ARG CR_TAG=23.13.26032.30
ARG IGC_TAG=igc-1.0.13700.14
ARG CM_TAG=latest
# level-zero v1.10.0+ depends on libstdc++6 (>= 11); however bullseye is based
# on gcc 10
# ARG L0_TAG=v1.9.9
ARG L0_TAG=latest

# ONEAPI
ARG ONEAPI_INSTALLER_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/7deeaac4-f605-4bcf-a81b-ea7531577c61
ARG ONEAPI_VERSION=2023.1.0.46401
ARG ONEAPI_INSTALL_BINARY_NAME=l_BaseKit_p_$ONEAPI_VERSION.sh
ARG ONEAPI_INSTALL_DIR=/opt/intel/oneapi

# Versions of the intel python packages
ARG DPCTL_GIT_BRANCH=0.14.4
ARG DPCTL_GIT_URL=https://github.com/IntelPython/dpctl.git

ARG DPNP_GIT_BRANCH=0.12.0
ARG DPNP_GIT_URL=https://github.com/IntelPython/dpnp.git

ARG DPCPP_LLVM_SPIRV_GIT_BRANCH=main
ARG DPCPP_LLVM_SPIRV_GIT_URL=https://github.com/IntelPython/dpcpp-llvm-spirv.git

ARG NUMBA_DPEX_GIT_BRANCH=0.22.0
ARG NUMBA_DPEX_GIT_URL=https://github.com/IntelPython/numba-dpex.git

# CMAKE
ARG CMAKE_VERSION=3.26
ARG CMAKE_VERSION_BUILD=4

# Python
ARG INTEL_NUMPY_VERSION="==1.24.3"
ARG INTEL_NUMBA_VERSION="==0.58"
ARG CYTHON_VERSION="==0.29.35"
ARG SCIKIT_BUILD_VERSION="==0.17.6"

# If you are have access to the internet via proxy.
# It is required for loading packages.
ARG http_proxy
ARG https_proxy

# Required to get drivers by github api. Just use github personal access token:
# https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
ARG GITHUB_USER=''
ARG GITHUB_PASSWORD=''

# Image names used in multistage build
ARG BASE_IMAGE=python:$PYTHON_VERSION-slim-bookworm
ARG RUNTIME_BASE_IMAGE=runtime-base
ARG BUILDER_IMAGE=builder
ARG DPCTL_BUILDER_IMAGE=dpctl-builder
ARG DPNP_BUILDER_IMAGE=dpnp-builder
ARG DPCPP_LLVM_SPIRV_BUILDER_IMAGE=dpcpp-llvm-spirv-builder
ARG NUMBA_DPEX_BUILDER_IMAGE=numba-dpex-builder
ARG TOOLKIT_IMAGE=toolkit
ARG NUMBA_DPEX_BUILDER_RUNTIME_IMAGE=numba-dpex-builder-runtime
ARG RUNTIME_IMAGE=runtime
ARG DRIVERS_IMAGE=drivers


FROM $BASE_IMAGE as base
ARG http_proxy
ARG https_proxy

SHELL ["/bin/bash", "-c"]

# Upgrade system to install latest packages
RUN \
    --mount=type=cache,target=/root/.cache/pip/ \
    export http_proxy=$http_proxy https_proxy=$https_proxy \
    && pip install --upgrade pip \
    && apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*


FROM base as oneapi
ARG ONEAPI_COMPONENTS="intel.oneapi.lin.dpcpp-cpp-compiler:intel.oneapi.lin.tbb.devel:intel.oneapi.lin.mkl.devel"
ARG ONEAPI_LOG_DIR=/tmp/intel/log
ARG ONEAPI_CACHE_DIR=/root/.cache/oneapi/cache
ARG ONEAPI_INSTALLER_CACHE_DIR=/root/.cache/oneapi/installer
ARG ONEAPI_INSTALLER_URL
ARG ONEAPI_INSTALL_BINARY_NAME
ARG ONEAPI_INSTALL_DIR=/opt/intel/oneapi
ARG http_proxy
ARG https_proxy
RUN \
    --mount=type=cache,target=/root/.cache/oneapi \
    export http_proxy=$http_proxy https_proxy=$https_proxy \
    && apt-get update && apt-get install -y wget \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p $ONEAPI_INSTALLER_CACHE_DIR \
    && cd $ONEAPI_INSTALLER_CACHE_DIR \
    && wget -nc -q $ONEAPI_INSTALLER_URL/$ONEAPI_INSTALL_BINARY_NAME  \
    && chmod +x $ONEAPI_INSTALL_BINARY_NAME \
    && ./$ONEAPI_INSTALL_BINARY_NAME -a -s --eula accept \
    --action install --components $ONEAPI_COMPONENTS  \
    --install-dir $ONEAPI_INSTALL_DIR --log-dir $ONEAPI_LOG_DIR \
    --download-cache $ONEAPI_CACHE_DIR

FROM base as toolkit
ARG ONEAPI_INSTALL_DIR=/opt/intel/oneapi
ARG TOOLKIT_DIR=/opt/toolkit
ARG http_proxy
ARG https_proxy

RUN \
    --mount=type=bind,target=$ONEAPI_INSTALL_DIR,source=$ONEAPI_INSTALL_DIR,from=oneapi \
    export http_proxy=$http_proxy https_proxy=$https_proxy \
    && apt-get update && apt-get install -y fdupes rsync procps coreutils \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/toolkit/lib \
    && mkdir -p /opt/toolkit/lib-dev \
    && mkdir -p /opt/toolkit/bin \
    && mkdir -p /opt/toolkit/bin-dev \
    && mkdir -p /opt/toolkit/include \
    && source $ONEAPI_INSTALL_DIR/setvars.sh \
    cd ${ONEAPI_INSTALL_DIR} \
    \
    && echo $LD_LIBRARY_PATH | tr ":" "\n" | grep mkl |  \
    xargs -i find {} -xtype f \
    | sort >> $TOOLKIT_DIR/lib-mkl-all.txt \
    && grep -v -E '\.a$' $TOOLKIT_DIR/lib-mkl-all.txt > $TOOLKIT_DIR/lib-mkl.txt \
    && comm -23 $TOOLKIT_DIR/lib-mkl-all.txt $TOOLKIT_DIR/lib-mkl.txt > $TOOLKIT_DIR/lib-mkl-dev.txt \
    && rm $TOOLKIT_DIR/lib-mkl-all.txt \
    \
    && echo $LD_LIBRARY_PATH | tr ":" "\n" | grep compiler  \
    | grep -v oclfpga \
    | xargs -i find {} -maxdepth 1 -xtype f \
    | sort >> $TOOLKIT_DIR/lib-compiler-all.txt \
    && grep -v -E '/compiler/lib/.*\.(a|dbg)$' $TOOLKIT_DIR/lib-compiler-all.txt > $TOOLKIT_DIR/lib-compiler.txt \
    && comm -23 $TOOLKIT_DIR/lib-compiler-all.txt $TOOLKIT_DIR/lib-compiler.txt > $TOOLKIT_DIR/lib-compiler-dev.txt \
    && rm $TOOLKIT_DIR/lib-compiler-all.txt \
    \
    && echo $LD_LIBRARY_PATH | tr ":" "\n" | grep tbb  \
    | xargs -i find {} -maxdepth 1 -xtype f \
    | grep -v _debug  \
    >> $TOOLKIT_DIR/lib-tbb.txt \
    \
    && echo $PATH | tr ":" "\n" \
    | grep -E "^$ONEAPI_INSTALL_DIR/(compiler|mkl)" \
    | grep -v '/lib/' \
    | xargs -i find {} -maxdepth 1 -xtype f \
    >> $TOOLKIT_DIR/bin-all.txt \
    \
    && find $ONEAPI_INSTALL_DIR/compiler/latest/linux/bin-llvm -maxdepth 1 -xtype f \
    >> $TOOLKIT_DIR/bin-all.txt \
    \
    && sort -o $TOOLKIT_DIR/bin-all.txt $TOOLKIT_DIR/bin-all.txt \
    && echo "bin-all.txt" \
    && cat $TOOLKIT_DIR/bin-all.txt \
    && cat $TOOLKIT_DIR/bin-all.txt | grep -E '/(sycl-ls|llvm-spirv)$' \
    >> $TOOLKIT_DIR/bin.txt \
    && comm -23 $TOOLKIT_DIR/bin-all.txt $TOOLKIT_DIR/bin.txt > $TOOLKIT_DIR/bin-dev.txt \
    && rm $TOOLKIT_DIR/bin-all.txt \
    \
    && find $ONEAPI_INSTALL_DIR -type d | grep -E '\/include$' \
    | grep -v oclfpga \
    | grep -v debugger \
    | xargs -i rsync -a {}/ $TOOLKIT_DIR/include/ \
    \
    && ls $TOOLKIT_DIR/lib*.txt | grep -v -E '\-dev\.txt$' | xargs cat \
    | xargs -i cp {} $TOOLKIT_DIR/lib/ \
    && cat $TOOLKIT_DIR/lib*-dev.txt \
    | xargs -i cp {} $TOOLKIT_DIR/lib-dev/ \
    && cat $TOOLKIT_DIR/bin.txt | xargs -i cp {} $TOOLKIT_DIR/bin/ \
    && cat $TOOLKIT_DIR/bin-dev.txt | xargs -i cp {} $TOOLKIT_DIR/bin-dev/ \
    `# removing duplicate libraries: broken packages?` \
    \
    && find $TOOLKIT_DIR -maxdepth 1 -type d \
    | grep lib \
    | xargs -i fdupes -qio name {} | \
    awk '{if ($0=="") ln=""; else if (ln=="") ln = $0; else system("rm " $0 ";\tln -sr " ln " " $0) }' \
    \
    && find $TOOLKIT_DIR -maxdepth 1 -type d \
    | grep bin \
    | xargs -i fdupes -qo name {} | \
    awk '{if ($0=="") ln=""; else if (ln=="") ln = $0; else system("rm " $0 ";\tln -sr " ln " " $0) }' \
    \
    && find /opt/toolkit/lib/ -type l -exec ln -sfr {} /opt/toolkit/lib/ \;


FROM $TOOLKIT_IMAGE as toolkit-dist


FROM base as runtime-base
RUN \
    --mount=type=bind,target=/opt/toolkit,source=/opt/toolkit,from=toolkit-dist \
    export http_proxy=$http_proxy https_proxy=$https_proxy \
    && apt-get update && apt-get install -y \
    spirv-headers \
    rsync \
    && rm -rf /var/lib/apt/lists/* \
    && rsync -a /opt/toolkit/bin/ /usr/local/bin/ \
    && rsync -a /opt/toolkit/lib/ /usr/local/lib/

ENV OCL_ICD_FILENAMES=libintelocl_emu.so:libintelocl.so


FROM $RUNTIME_BASE_IMAGE as builder-base
RUN \
    --mount=type=bind,target=/opt/toolkit,source=/opt/toolkit,from=toolkit-dist \
    rsync -a /opt/toolkit/bin-dev/ /usr/local/bin/ \
    && rsync -a /opt/toolkit/lib-dev/ /usr/local/lib/ \
    && rsync -a /opt/toolkit/include/ /usr/local/include/


FROM base as drivers
ARG CR_TAG
ARG IGC_TAG
ARG CM_TAG
ARG L0_TAG
ARG DRIVER_CACHE_DIR=/root/.cache/drivers
ARG GITHUB_USER
ARG GITHUB_PASSWORD
ARG http_proxy
ARG https_proxy

COPY /scripts/github_load.py /opt/github_load.py

RUN \
    --mount=type=cache,target=$DRIVER_CACHE_DIR \
    export http_proxy=$http_proxy https_proxy=$https_proxy \
    && python /opt/github_load.py -u $GITHUB_USER -p $GITHUB_PASSWORD -c $DRIVER_CACHE_DIR \
    intel/intel-graphics-compiler -t $IGC_TAG -d /opt/install/graphics/ -g '.*deb' \
    && python /opt/github_load.py -u $GITHUB_USER -p $GITHUB_PASSWORD -c $DRIVER_CACHE_DIR \
    intel/compute-runtime -t $CR_TAG -d /opt/install/graphics/ -g '.*deb' \
    && python /opt/github_load.py -u $GITHUB_USER -p $GITHUB_PASSWORD -c $DRIVER_CACHE_DIR \
    intel/cm-compiler -t $CM_TAG -d /opt/install/graphics/ -g '.*u20.*deb' \
    && python /opt/github_load.py -u $GITHUB_USER -p $GITHUB_PASSWORD -c $DRIVER_CACHE_DIR \
    oneapi-src/level-zero -t $L0_TAG -d /opt/install/graphics/ -g '.*deb'


ARG DRIVERS_IMAGE

FROM $DRIVERS_IMAGE as drivers-dist


FROM builder-base as builder
ARG CMAKE_VERSION
ARG CMAKE_VERSION_BUILD
ARG CMAKE_BUILD_DIR=/build/cmake
ARG CMAKE_DOWNLOAD_DIR=/root/.cache/cmake
ARG CMAKE_INSTALL_DIR=/usr/local
ARG CMAKE_INSTALLER_NAME=cmake-${CMAKE_VERSION}.${CMAKE_VERSION_BUILD}-linux-x86_64.sh
ARG INTEL_NUMBA_VERSION
ARG INTEL_NUMPY_VERSION
ARG WHEEL_VERSION
ARG SCIKIT_BUILD_VERSION
ARG http_proxy
ARG https_proxy

# Installing building packages
RUN \
    export http_proxy=$http_proxy https_proxy=$https_proxy \
    && apt-get update && apt-get install -y \
    wget \
    build-essential \
    git \
    ninja-build \
    procps `# tbb runtime` \
    ocl-icd-libopencl1 `# tbb runtime?` \
    && apt-get remove -y cmake \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Installing CMake
RUN \
    --mount=type=cache,target=$CMAKE_DOWNLOAD_DIR \
    export http_proxy=$http_proxy https_proxy=$https_proxy \
    && mkdir -p $CMAKE_BUILD_DIR $CMAKE_INSTALL_DIR \
    && cd $CMAKE_DOWNLOAD_DIR \
    && wget -nc -q https://cmake.org/files/v${CMAKE_VERSION}/${CMAKE_INSTALLER_NAME} \
    && cd $CMAKE_BUILD_DIR \
    && sh ${CMAKE_DOWNLOAD_DIR}/${CMAKE_INSTALLER_NAME} \
    --prefix=${CMAKE_INSTALL_DIR} --skip-license \
    && rm -rf $CMAKE_BUILD_DIR

WORKDIR /build

# Install python dependencies
RUN \
    --mount=type=cache,target=/root/.cache/pip/ \
    export http_proxy=$http_proxy https_proxy=$https_proxy \
    && pip install -U \
    numba${INTEL_NUMBA_VERSION} \
    numpy${INTEL_NUMPY_VERSION} \
    cython${CYTHON_VERSION} \
    scikit-build${SCIKIT_BUILD_VERSION}


FROM $BUILDER_IMAGE AS dpctl-builder
ARG DPCTL_GIT_BRANCH
ARG DPCTL_GIT_URL
ARG DPCTL_BUILD_DIR=/build
ARG DPCTL_DIST_DIR=/dist
ARG SKBUILD_ARGS="-- -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icpx"
ARG http_proxy
ARG https_proxy

RUN \
  export http_proxy=$http_proxy https_proxy=$https_proxy \
  && mkdir -p $DPCTL_BUILD_DIR \
  && mkdir $DPCTL_DIST_DIR \
  && cd $DPCTL_BUILD_DIR \
  && git clone --recursive -b $DPCTL_GIT_BRANCH --depth 1 $DPCTL_GIT_URL . \
  && python setup.py bdist_wheel ${SKBUILD_ARGS} \
  && cp dist/dpctl*.whl $DPCTL_DIST_DIR


FROM $DPCTL_BUILDER_IMAGE AS dpctl-builder-dist


FROM $BUILDER_IMAGE AS dpnp-builder
ARG DPNP_BUILD_DIR=/build
ARG DPNP_DIST_DIR=/dist
ARG DPNP_GIT_BRANCH
ARG DPNP_GIT_URL
ARG SKBUILD_ARGS="-- -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icpx"
ARG http_proxy
ARG https_proxy

RUN \
    --mount=type=bind,target=/mnt/dpctl,source=/dist,from=dpctl-builder-dist \
    --mount=type=cache,target=/root/.cache/pip/ \
    export http_proxy=$http_proxy https_proxy=$https_proxy \
    && pip install -U /mnt/dpctl/dpctl*.whl \
    && mkdir -p $DPNP_BUILD_DIR \
    && mkdir -p $DPNP_DIST_DIR \
    && cd $DPNP_BUILD_DIR \
    && git clone --recursive -b $DPNP_GIT_BRANCH --depth 1 $DPNP_GIT_URL . \
    && export DPCTL_MODULE_PATH=$(python -m dpctl --cmakedir) \
    && export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$DPCTL_MODULE_PATH \
    # && python setup.py build_clib ${SKBUILD_ARGS} \
    # # && python setup.py build_ext ${SKBUILD_ARGS} \
    && python setup.py bdist_wheel ${SKBUILD_ARGS} \
    && cp dist/dpnp*.whl $DPNP_DIST_DIR


FROM $DPNP_BUILDER_IMAGE AS dpnp-builder-dist


FROM $BUILDER_IMAGE AS dpcpp-llvm-spirv-builder

ARG ONEAPI_INSTALL_DIR
ARG DPCPP_LLVM_SPIRV_GIT_BRANCH
ARG DPCPP_LLVM_SPIRV_GIT_URL
ARG DPCPP_LLVM_SPIRV_BUILD_DIR=/build
ARG DPCPP_LLVM_SPIRV_DIST_DIR=/dist
ARG http_proxy
ARG https_proxy

RUN \
  export http_proxy=$http_proxy https_proxy=$https_proxy \
  && mkdir -p $DPCPP_LLVM_SPIRV_BUILD_DIR \
  && mkdir $DPCPP_LLVM_SPIRV_DIST_DIR \
  && cd $DPCPP_LLVM_SPIRV_BUILD_DIR \
  && cd $DPCPP_LLVM_SPIRV_BUILD_DIR \
  && git clone --recursive -b $DPCPP_LLVM_SPIRV_GIT_BRANCH --depth 1 $DPCPP_LLVM_SPIRV_GIT_URL . \
  && cd pkg \
  && python setup.py bdist_wheel \
  && cp dist/dpcpp_llvm_spirv*.whl $DPCPP_LLVM_SPIRV_DIST_DIR


FROM $DPCPP_LLVM_SPIRV_BUILDER_IMAGE AS dpcpp-llvm-spirv-dist


FROM $BUILDER_IMAGE AS numba-dpex-builder-runtime
ARG ONEAPI_INSTALL_DIR
ARG NUMBA_DPEX_BUILD_DIR=/build
ARG NUMBA_DPEX_DIST_DIR=/dist
ARG NUMBA_DPEX_GIT_BRANCH
ARG NUMBA_DPEX_GIT_URL
ARG INTEL_PYPI_URL
ARG BASE_PYPI_URL
ARG http_proxy
ARG https_proxy

RUN \
  --mount=type=bind,target=/mnt/dpctl,source=/dist,from=dpctl-builder-dist \
  --mount=type=bind,target=/mnt/dpnp,source=/dist,from=dpnp-builder-dist \
  --mount=type=bind,target=/mnt/dpcpp_llvm_spirv,source=/dist,from=dpcpp-llvm-spirv-dist \
  --mount=type=cache,target=/root/.cache/pip/ \
  export http_proxy=$http_proxy https_proxy=$https_proxy \
  && pip install -U \
  /mnt/dpctl/dpctl*.whl /mnt/dpnp/dpnp*.whl \
  /mnt/dpcpp_llvm_spirv/dpcpp_llvm_spirv*.whl \
  && ln -s /usr/local/bin/llvm-spirv /usr/local/lib/python*/site-packages/dpcpp_llvm_spirv/ \
  && mkdir -p $NUMBA_DPEX_BUILD_DIR \
  && mkdir $NUMBA_DPEX_DIST_DIR \
  && cd $NUMBA_DPEX_BUILD_DIR \
  && git clone --recursive -b $NUMBA_DPEX_GIT_BRANCH --depth 1 $NUMBA_DPEX_GIT_URL .

FROM $NUMBA_DPEX_BUILDER_RUNTIME_IMAGE AS numba-dpex-builder
ARG NUMBA_DPEX_DIST_DIR=/dist

RUN \
    # HACK: currently, there is an issue with the bdist_wheel configuration for numba_dpex
    # that causes missing files in the final tarball.
    # The workaround consists in triggering building steps by running `setup.py develop`
    # before running `setup.py bdist_wheel`.
    # See https://github.com/soda-inria/sklearn-numba-dpex/issues/5
    python setup.py develop \
    # XXX: is it needed to pass manylinux wheel build arg to the setup command ?
    && python setup.py bdist_wheel \
    && cp dist/numba_dpex*.whl $NUMBA_DPEX_DIST_DIR


FROM $RUNTIME_BASE_IMAGE as runtime
ARG ONEAPI_INSTALL_DIR
ARG INTEL_NUMPY_VERSION
ARG INTEL_NUMBA_VERSION
ARG ONEAPI_VERSION
# ARG USERNAME=numba_dpex
# ARG USER_UID=1000
# ARG USER_GID=$USER_UID
ARG http_proxy
ARG https_proxy

# Package dependencies
RUN \
    export http_proxy=$http_proxy https_proxy=$https_proxy \
    && apt-get update && apt-get install -y \
    ocl-icd-libopencl1 `# gpu runtime` \
    # procps `# tbb runtime` \
    gcc g++ `# dpctl runtime` \
    fdupes `# remove duplicate libraries installed by pip` \
    && rm -rf /var/lib/apt/lists/*

# DPNP does not ship tests with package so we deliver it here to be able to test environment
COPY --from=dpnp-builder-dist /build/tests /opt/dpnp/tests

# runtime python packages
RUN \
  --mount=type=bind,target=/mnt/dpctl,source=/dist,from=dpctl-builder-dist \
  --mount=type=bind,target=/mnt/dpnp,source=/dist,from=dpnp-builder-dist \
  --mount=type=bind,target=/mnt/dpcpp_llvm_spirv,source=/dist,from=dpcpp-llvm-spirv-dist \
  --mount=type=bind,target=/mnt/numba_dpex,source=/dist,from=numba-dpex-builder \
  --mount=type=cache,target=/root/.cache/pip/ \
  export http_proxy=$http_proxy https_proxy=$https_proxy \
  && pip install -U \
  numpy${INTEL_NUMPY_VERSION} \
  cython${CYTHON_VERSION} \
  numba${INTEL_NUMBA_VERSION} \
  /mnt/dpctl/dpctl*.whl \
  /mnt/dpnp/dpnp*.whl \
  /mnt/dpcpp_llvm_spirv/dpcpp_llvm_spirv*.whl \
  /mnt/numba_dpex/numba_dpex*.whl \
  && ln -s /usr/local/bin/llvm-spirv /usr/local/lib/python*/site-packages/dpcpp_llvm_spirv/ \
  && fdupes -qio name /usr/local/lib/python*/site-packages/dpctl/ | \
  awk '{if ($0=="") ln=""; else if (ln=="") ln = $0; else system("rm " $0 ";\tln -s " ln " " $0) }'

# Create an user
# TODO: there is no access to gpu with non root user. Same issue on intel/llvm docker.
# RUN groupadd --gid $USER_GID $USERNAME \
#   && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
# USER $USERNAME
# WORKDIR /home/$USERNAME


FROM $RUNTIME_IMAGE as runtime-gpu
ARG http_proxy
ARG https_proxy

RUN \
    export http_proxy=$http_proxy https_proxy=$https_proxy \
    && apt-get update && apt-get install -y \
    ocl-icd-libopencl1 `# gpu runtime` \
    && rm -rf /var/lib/apt/lists/*

# Drivers setup
RUN --mount=type=bind,target=/mnt/opt,source=/opt,from=drivers-dist \
    cd /mnt/opt/install/graphics && dpkg -i *.deb && dpkg -i *.ddeb


FROM $NUMBA_DPEX_BUILDER_RUNTIME_IMAGE AS numba-dpex-builder-runtime-gpu

# Drivers setup
RUN --mount=type=bind,target=/mnt/opt,source=/opt,from=drivers-dist \
    cd /mnt/opt/install/graphics && dpkg -i *.deb && dpkg -i *.ddeb
