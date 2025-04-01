#pragma once

#include <algorithm>
#include <cassert>
#include <queue>
#include <thread>
#include <vector>
#include "space/i16.hpp"

struct GreedyNet {
    using space_t = space::I16;
    using dist_t = space_t::dist_t;
    const space_t& space;

    const size_t n;
    std::vector<std::vector<size_t>> go;

    size_t shard_count, pool_size;
    std::vector<size_t> shard_cuts;
    std::vector<uint8_t> used;
    size_t eval_dist;
    uint8_t utimer;

    GreedyNet(const space_t& space);

    // Time complexity: O(n^2) calculating distances
    void Build();

    std::pair<dist_t, size_t> SearchInShard(space_t::computer_t& comp,
                                            size_t sid);

    std::pair<dist_t, size_t> SearchInShardWithPool(space_t::computer_t comp,
                                                    size_t sid);
    // Search k nearest points to point q
    std::vector<size_t> Search(space_t::point_t q, size_t k);
};