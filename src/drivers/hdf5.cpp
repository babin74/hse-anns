#include <drivers/hdf5.hpp>

#include <H5Dpublic.h>
#include <H5Ipublic.h>
#include <H5Ppublic.h>
#include <H5Spublic.h>
#include <H5Tpublic.h>
#include <hdf5.h>

#include <array>
#include <climits>
#include <iostream>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

struct TableReadResult {
    hid_t dset;
    hid_t fspace;
    hsize_t n, m;
};

TableReadResult ReadHeadingTable(hid_t fd, const std::string_view name) {
    TableReadResult res{
        .dset = H5I_INVALID_HID,
        .fspace = H5I_INVALID_HID,
        .n = 0,
        .m = 0,
    };

    hid_t dset = H5Dopen(fd, name.data(), H5P_DEFAULT);
    if (dset == H5I_INVALID_HID) {
        std::cerr << "Unable to find dataset `" << name << "`" << std::endl;
        return res;
    }

    hid_t space = H5Dget_space(dset);
    if (space == H5I_INVALID_HID) {
        std::cerr << "Unable to gen data space" << std::endl;
        H5Dclose(dset);
        return res;
    }

    hsize_t rank = H5Sget_simple_extent_ndims(space);
    if (rank != static_cast<hsize_t>(2)) {
        std::cerr << "Table expected, rank = " << rank << " != " << 2
                  << std::endl;
        H5Sclose(space);
        H5Dclose(dset);
        return res;
    }

    std::array<hsize_t, 2> cur_dims, max_dims;
    H5Sget_simple_extent_dims(space, cur_dims.data(), max_dims.data());
    std::ignore = max_dims;

    res.dset = dset;
    res.fspace = space;
    res.n = cur_dims[0];
    res.m = cur_dims[1];
    return res;
}

}  // namespace

Hdf5Reader Hdf5Reader::Hold(hid_t fd) {
    return Hdf5Reader(fd);
}

Hdf5Reader Hdf5Reader::Open(std::string_view filename) {
    hid_t fd = H5Fopen(filename.data(), H5F_ACC_RDONLY, H5P_DEFAULT);
    return Hdf5Reader(fd);
}

Hdf5Reader::Hdf5Reader(hid_t fd) : file_id_(fd) {}

Hdf5Reader::~Hdf5Reader() {
    H5Fclose(file_id_);
}

space::I16 Hdf5Reader::ReadI16(std::string_view dset_name) {
    space::I16 res;
    std::cerr << "Try to read " << dset_name << " [I16]..." << std::flush;

    auto [dset, space, n, m] = ReadHeadingTable(file_id_, dset_name);
    std::cerr << " --> size: " << n << "x" << m << std::endl;

    const hsize_t rows_cache = (m - 1 + (1 << 10)) / m;

    res.data.resize(n * m);
    herr_t status = H5Dread(dset, H5T_NATIVE_INT16, H5S_ALL, space, H5P_DEFAULT,
                            res.data.data());
    if (status < 0) {
        std::cerr << "Error reading" << std::endl;
    }

    res.dim = m;

    H5Sclose(space);
    H5Dclose(dset);
    return res;
}