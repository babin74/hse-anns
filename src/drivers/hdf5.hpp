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

    template <space::Space S>
    S ReadSpace();

    template <>
    space::I16 ReadSpace<space::I16>() {
        return _ReadSpaceI16();
    }

    template <>
    space::I32 ReadSpace<space::I32>() {
        return _ReadSpaceI32();
    }

   private:
    Hdf5Reader(hid_t fd);

    space::I16 _ReadSpaceI16();
    space::I32 _ReadSpaceI32();
};