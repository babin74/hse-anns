#pragma once

#include <algorithm>
#include <cassert>
#include <set>
#include <vector>
#include "space/i16.hpp"

struct GreedyNet {
    using space_t = space::I16;
    using dist_t = space_t::dist_t;
    const space_t& space;

    const size_t n;
    std::vector<std::vector<size_t>> go;

    size_t shard_count;
    std::vector<size_t> shard_cuts;
    std::vector<size_t> neighbours_;

    GreedyNet(const space_t& space) : space(space), n(space.Size()), go(n) {
        shard_count = 256;
    }

    // Time complexity: O(n^2) calculating distances
    void Build() {
        for (size_t i = 0; i <= shard_count; ++i)
            shard_cuts.push_back(i * n / shard_count);

        std::vector<size_t> last_upd(n, 1);
        for (size_t i = 0; i < n; ++i) {
            auto calc = space.GetComputer(i);
            for (size_t sid = 0; sid < shard_count; ++sid) {
                size_t jbest = shard_cuts[sid];
                dist_t dbest = calc.Distance(jbest);
                for (size_t j = jbest + 1; j < shard_cuts[sid + 1]; ++j) {
                    dist_t d = calc.Distance(j);
                    if (dbest > d) {
                        dbest = d;
                        go[jbest].push_back(j);
                        if (go[jbest].size() >= last_upd[jbest] * 2) {
                            std::ranges::sort(go[jbest]);
                            go[jbest].resize(std::unique(go[jbest].begin(),
                                                         go[jbest].end()) -
                                             go[jbest].begin());
                            last_upd[jbest] = go[jbest].size();
                        }
                        jbest = j;
                    }
                }
            }
        }

        for (size_t i = 0; i < n; ++i) {
            std::ranges::sort(go[i]);
            go[i].resize(std::unique(go[i].begin(), go[i].end()) -
                         go[i].begin());
        }
    }

    std::pair<dist_t, size_t> SearchInShard(space_t::computer_t& comp,
                                            size_t sid) {
        std::set<size_t> used;
        size_t jbest = shard_cuts[sid];
        dist_t dbest = comp.Distance(jbest);
        while (true) {
            bool any = false;
            for (size_t j : go[jbest]) {
                if (!used.insert(j).second)
                    continue;
                dist_t d = comp.Distance(j);
                if (dbest > d) {
                    dbest = d;
                    jbest = j;
                    any = true;
                    break;
                }
            }

            if (!any)
                return {dbest, jbest};
        }
    }

    // Search k nearest points to point q
    std::vector<size_t> Search(space_t::point_t q, size_t k) {
        assert(k == 1);
        auto comp = space.GetComputer(q);

        using cand_t = std::pair<dist_t, size_t>;
        std::vector<cand_t> cands(shard_count);
        for (size_t sid = 0; sid < shard_count; ++sid) {
            cands[sid] = SearchInShard(comp, sid);
        }

        size_t ans = std::min_element(cands.begin(), cands.end())->second;
        return {ans};
    }
};