if [ ! -d "$HOME/menoh${MENOH_INSALL_SUFFIX}/lib" ]; then
    if [ "$TRAVIS_OS_NAME" = "linux" ]; then export CXX="g++-7" CC="gcc-7"; fi
    git clone https://github.com/pfnet-research/menoh.git
    cd menoh
    git checkout $MENOH_REV
    mkdir -p build
    cd build
    cmake \
      -DMKLDNN_INCLUDE_DIR="$HOME/mkl-dnn${MKL_DNN_INSALL_SUFFIX}/include" \
      -DMKLDNN_LIBRARY="$HOME/mkl-dnn${MKL_DNN_INSALL_SUFFIX}/lib/libmkldnn.so" \
      -DCMAKE_INSTALL_PREFIX=$HOME/menoh${MENOH_INSALL_SUFFIX} \
      ..
    make
    make install
    cd ../..
else
    echo "Using cached directory."
fi
