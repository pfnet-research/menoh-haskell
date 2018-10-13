if [ ! -d "$HOME/menoh${MENOH_INSTALL_SUFFIX}/lib" ]; then
    if [ "$TRAVIS_OS_NAME" = "linux" ]; then export CXX="g++-7" CC="gcc-7"; fi
    git clone https://github.com/pfnet-research/menoh.git --recurse-submodules
    cd menoh
    git checkout $MENOH_REV
    mkdir -p build
    cd build
    if [ "$TRAVIS_OS_NAME" = "linux" ]; then
      cmake \
        -DENABLE_TEST=OFF -DENABLE_BENCHMARK=OFF -DENABLE_EXAMPLE=OFF \
        -DMKLDNN_INCLUDE_DIR="$HOME/mkl-dnn${MKL_DNN_INSTALL_SUFFIX}/include" \
        -DMKLDNN_LIBRARY="$HOME/mkl-dnn${MKL_DNN_INSTALL_SUFFIX}/lib/libmkldnn.so" \
        -DCMAKE_INSTALL_PREFIX=$HOME/menoh${MENOH_INSTALL_SUFFIX} \
        ..
    else
      cmake \
        -DENABLE_TEST=OFF -DENABLE_BENCHMARK=OFF -DENABLE_EXAMPLE=OFF \
        -DCMAKE_INSTALL_PREFIX=$HOME/menoh${MENOH_INSTALL_SUFFIX} \
        -DCMAKE_INSTALL_NAME_DIR=$HOME/menoh${MENOH_INSTALL_SUFFIX}/lib \
        ..
    fi
    make
    make install
    cd ../..
else
    echo "Using cached directory."
fi
