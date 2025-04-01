#include "index/greedy-net.hpp"
#include "space/i16.hpp"

using dist_t = GreedyNet::dist_t;

GreedyNet::GreedyNet(const space_t& space)
    : space(space),
      n(space.Size()),
      go(n),
      shard_count(1),
      pool_size(0),
      shard_cuts(),
      used(n, 0),
      eval_dist(0),
      utimer(0) {
    // used.assign(n, 0);
    // utimer = 0;
}

namespace {
void BuildLayer(GreedyNet* net, size_t sid, size_t* last_upd) {
    for (size_t i = 0; i < net->n; ++i) {
        auto calc = net->space.GetComputer(i);
        size_t jbest = net->shard_cuts[sid];
        dist_t dbest = calc.Distance(jbest);
        for (size_t j = jbest + 1; j < net->shard_cuts[sid + 1]; ++j) {
            dist_t d = calc.Distance(j);
            if (dbest > d) {
                dbest = d;
                net->go[jbest].push_back(j);
                if (net->go[jbest].size() >= last_upd[jbest] * 2) {
                    std::ranges::sort(net->go[jbest]);
                    net->go[jbest].resize(std::unique(net->go[jbest].begin(),
                                                      net->go[jbest].end()) -
                                          net->go[jbest].begin());
                    last_upd[jbest] = net->go[jbest].size();
                }
                jbest = j;
            }
        }
    }
}
}  // namespace

// Time complexity: O(n^2) calculating distances
void GreedyNet::Build() {
    for (size_t i = 0; i <= shard_count; ++i)
        shard_cuts.push_back(i * n / shard_count);

    std::vector<size_t> last_upd(n, 1);
    std::vector<std::thread> jobs(shard_count);

    for (size_t sid = 0; sid < shard_count; ++sid) {
        jobs[sid] = std::thread(BuildLayer, this, sid, last_upd.data());
    }
    for (size_t sid = 0; sid < shard_count; ++sid) {
        jobs[sid].join();
    }

    for (size_t i = 0; i < n; ++i) {
        std::ranges::sort(go[i]);
        go[i].resize(std::unique(go[i].begin(), go[i].end()) - go[i].begin());
    }
}

std::pair<dist_t, size_t> GreedyNet::SearchInShard(space_t::computer_t& comp,
                                                   size_t sid) {
    size_t jbest = shard_cuts[sid];
    dist_t dbest = comp.Distance(jbest);
    ++eval_dist;
    while (true) {
        bool any = false;
        for (size_t j : go[jbest]) {
            if (used[j] == utimer)
                continue;
            used[j] = utimer;

            dist_t d = comp.Distance(j);
            ++eval_dist;
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

std::pair<dist_t, size_t> GreedyNet::SearchInShardWithPool(
    space_t::computer_t comp, size_t sid) {
    if (pool_size == 0)
        return SearchInShard(comp, sid);

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
            if (used[j] == utimer)
                continue;

            const dist_t d = comp.Distance(j);
            eval_dist++;
            if (d < cur_dist) {
                que.emplace(d, j, 0);
                used[j] = utimer;
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
std::vector<size_t> GreedyNet::Search(space_t::point_t q, size_t k) {
    ++utimer;
    if (utimer == 0) {
        std::fill(used.begin(), used.end(), 0);
        ++utimer;
    }

    eval_dist = 0;
    assert(k == 1);
    auto comp = space.GetComputer(q);
    using comp_t = space::I16::computer_t;
    using cand_t = std::pair<dist_t, size_t>;
    std::vector<cand_t> cands(shard_count);
    for (size_t sid = 0; sid < shard_count; ++sid)
        cands[sid] = SearchInShardWithPool(comp, sid);

    size_t ans = std::min_element(cands.begin(), cands.end())->second;
    return {ans};
}