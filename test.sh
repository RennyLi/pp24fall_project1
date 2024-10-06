cd build
make clean
make -j
cd ..
./build/src/gpu/openacc_PartC ./images/4K-RGB.jpg ./output/4K-Bilateral-openacc.jpg