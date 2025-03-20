#pragma once

#include <H5Ipublic.h>
#include <space/i16.hpp>
#include <space/i32.hpp>
#include <string_view>

class Hdf5Reader final {
   public:
    static Hdf5Reader Open(const std::string_view filename);
    static Hdf5Reader Hold(hid_t fd);
    ~Hdf5Reader();

    space::I16 ReadI16(std::string_view dset_name);

   private:
    hid_t file_id_;

    explicit Hdf5Reader(hid_t fd);
};
