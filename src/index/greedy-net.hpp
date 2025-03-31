#pragma once

#include <algorithm>
#include <cassert>
#include <queue>
#include <set>
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
    std::vector<size_t> neighbours_;
    size_t eval_dist;

    GreedyNet(const space_t& space) : space(space), n(space.Size()), go(n) {
        shard_count = 256;
        pool_size = 0;
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
        eval_dist++;
        while (true) {
            bool any = false;
            for (size_t j : go[jbest]) {
                if (!used.insert(j).second)
                    continue;
                dist_t d = comp.Distance(j);
                eval_dist++;
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

    std::pair<dist_t, size_t> SearchInShardWithPool(space_t::computer_t& comp,
                                                    size_t sid) {
        if (pool_size == 0)
            return SearchInShard(comp, sid);

        std::set<size_t> used;
        std::priority_queue<std::tuple<dist_t, size_t, size_t>> que;
        size_t jbest = shard_cuts[sid];
        dist_t dbest = comp.Distance(jbest);
        eval_dist++;
        que.emplace(dbest, jbest, 0);
        int cnt = 0;
        while (!que.empty()) {
            bool any = false;
            auto [cur_dist, cur, ptr] = que.top();
            que.pop();

            if (cur_dist < dbest) {
                dbest = cur_dist;
                jbest = cur;
            }

            while (ptr != go[cur].size()) {
                const size_t j = go[cur][ptr++];
                if (used.count(j))
                    continue;

                const dist_t d = comp.Distance(j);
                eval_dist++;
                if (d < cur_dist) {
                    que.emplace(d, j, 0);
                    used.insert(j);
                    any = true;

                    if (que.size() > pool_size)
                        que.pop();
                    break;
                }
            }

            if (any) {
                que.emplace(cur_dist, cur, ptr);
                if (que.size() > pool_size)
                    que.pop();
            }
        }

        return {dbest, jbest};
    }

    // Search k nearest points to point q
    std::vector<size_t> Search(space_t::point_t q, size_t k) {
        eval_dist = 0;
        assert(k == 1);
        auto comp = space.GetComputer(q);

        using cand_t = std::pair<dist_t, size_t>;
        std::vector<cand_t> cands(shard_count);
        for (size_t sid = 0; sid < shard_count; ++sid) {
            cands[sid] = SearchInShardWithPool(comp, sid);
        }

        size_t ans = std::min_element(cands.begin(), cands.end())->second;
        return {ans};
    }
};