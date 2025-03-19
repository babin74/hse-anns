#include <H5Spublic.h>
#include <hdf5.h>

#include "space/i16.hpp"
#include "drivers/hdf5.hpp"

const char* dataset_dirname = "../datasets/siftsmall-128-euclidean.hdf5";

int main() {  
    auto reader = Hdf5Reader::Open(dataset_dirname);
    auto space = reader.ReadSpace<space::I16>();
}