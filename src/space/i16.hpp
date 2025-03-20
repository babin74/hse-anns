#pragma once

#include <cstdint>
#include <vector>

#include "space.hpp"

namespace space {

struct I16 {
    size_t dim;
    std::vector<int16_t> data;

    struct Computer {
        using dist_t = int32_t;

        size_t dim;
        const int16_t* data;
        const int16_t* me;

        inline dist_t Distance(size_t idx) const {
            dist_t res = 0;

            const int16_t* other = data + idx * dim;
            for (size_t i = dim; i--;) {
                int32_t dif = other[i] - me[i];
                res += dif * dif;
            }
            return res;
        };
    };

    inline size_t Size() const { return data.size() / dim; }

    inline Computer GetComputer(size_t idx) {
        return Computer{
            .dim = dim, .data = data.data(), .me = data.data() + dim * idx};
    }

    using point_t = const int16_t*;
    inline Computer GetComputer(point_t q) {
        return Computer{.dim = dim, .data = data.data(), .me = q};
    }

    inline point_t GetPoint(size_t idx) {
        return data.data() + dim * idx;
    }

    using computer_t = Computer;
    using dist_t = computer_t::dist_t;
};

static_assert(Space<I16>);

}  // namespace space