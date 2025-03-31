#pragma once

#include <chrono>
#include <iostream>

namespace task {

struct BuildAndTestResult {
    int64_t build_time;
    int64_t search_time;
    std::vector<std::vector<size_t>> ans;
};

struct RunResult {
    int64_t search_time;
    std::vector<std::vector<size_t>> ans;
};

template <typename Searcher, typename Space>
inline RunResult JustRun(Searcher& searcher, const Space& test,
                                       size_t k, bool verbose = true) {
    using hc = std::chrono::high_resolution_clock;
    RunResult res;
    res.ans.resize(test.Size());

    auto ts_search_time = hc::now();
    for (size_t i = 0; i < test.Size(); ++i) {
        auto q = test.GetPoint(i);
        res.ans[i] = searcher.Search(q, k);
    }
    res.search_time = (hc::now() - ts_search_time).count();
    if (verbose) {
        std::cerr << "Query time: " << res.search_time << " ns" << std::endl;
    }

    return res;
}

template <typename Searcher, typename Space>
inline BuildAndTestResult BuildAndTest(const Space& data, const Space& test,
                                       size_t k, bool verbose = true) {

    using hc = std::chrono::high_resolution_clock;
    BuildAndTestResult res;

    auto ts_build_start = hc::now();
    Searcher searcher(data);
    searcher.Build();
    res.build_time = (hc::now() - ts_build_start).count();
    if (verbose) {
        std::cerr << "Build time: " << res.build_time << " ns" << std::endl;
    }

    res.ans.resize(test.Size());

    auto ts_search_time = hc::now();
    for (size_t i = 0; i < test.Size(); ++i) {
        auto q = test.GetPoint(i);
        res.ans[i] = searcher.Search(q, k);
    }
    res.search_time = (hc::now() - ts_search_time).count();
    if (verbose) {
        std::cerr << "Query time: " << res.search_time << " ns" << std::endl;
    }

    return res;
}

template <typename Searcher, typename Space>
inline BuildAndTestResult BuildAndTest(Searcher& searcher, const Space& test,
                                       size_t k, bool verbose = true) {

    using hc = std::chrono::high_resolution_clock;
    BuildAndTestResult res;

    auto ts_build_start = hc::now();
    searcher.Build();
    res.build_time = (hc::now() - ts_build_start).count();
    if (verbose) {
        std::cerr << "Build time: " << res.build_time << " ns" << std::endl;
    }

    res.ans.resize(test.Size());

    auto ts_search_time = hc::now();
    for (size_t i = 0; i < test.Size(); ++i) {
        auto q = test.GetPoint(i);
        res.ans[i] = searcher.Search(q, k);
    }
    res.search_time = (hc::now() - ts_search_time).count();
    if (verbose) {
        std::cerr << "Query time: " << res.search_time << " ns" << std::endl;
    }

    return res;
}

int TestGreedyNetParams(int argc, char** argv);

}  // namespace task