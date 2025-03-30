#pragma once

#include <algorithm>
#include <vector>
#include "space/i16.hpp"

struct LinearIndex {
    using space_t = space::I16;
    using dist_t = space_t::dist_t;
    const space_t& space;

    LinearIndex(const space_t& space) : space(space) {}

    void Build() {}

    // Search k nearest points to point q
    std::vector<size_t> Search(space_t::point_t q, size_t k) {
        std::vector<std::pair<dist_t, size_t>> res;
        res.reserve(2 * k);
        auto comp = space.GetComputer(q);
        for (size_t i = 0; i < space.Size(); ++i) {
            res.emplace_back(comp.Distance(i), i);
            if (res.size() == 2 * k) {
                std::nth_element(res.begin(), res.begin() + k, res.end());
                res.resize(k);
            }
        }

        if (res.size() > k) {
            std::nth_element(res.begin(), res.begin()+k, res.end());
            res.resize(k);
        }

        std::sort(res.begin(), res.end());
        std::vector<size_t> ans(res.size());
        for (size_t i = 0; i != res.size(); ++i)
            ans[i] = res[i].second;
        return ans;
    }
};