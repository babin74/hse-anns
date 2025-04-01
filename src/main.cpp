#include <chrono>
#include <drivers/hdf5.hpp>
#include <iomanip>
#include <limits>
#include <space/i16.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string_view>
#include <thread>
#include <vector>
#include "index/greedy-net.hpp"
#include "index/hnsw.hpp"
#include "index/linear.hpp"
#include "task/build_and_test.hpp"

const char* dataset_dirname = "../datasets/siftsmall-128-euclidean.hdf5";

int main(int argc, char** argv) {
    if (argc > 1) {
        if (std::string_view(argv[1]) == "test") {
            assert(argc > 2);
            if (std::string_view(argv[2]) == "greedy-net") {
                return task::TestGreedyNetParams(argc, argv);
            }
            if (std::string_view(argv[2]) == "hnsw") {
                return task::TestHnswParams(argc, argv);
            }
        }
    }

    // return task::TestGreedyNetParams(argc, argv);

    const int shard_count = 16;
    const int pool_size = 4;

    auto reader = Hdf5Reader::Open(dataset_dirname);
    auto space_test = reader.ReadI16("/test");
    auto space_train = reader.ReadI16("/train");

    const int dim = space_test.dim;
    assert(dim == space_test.dim);
    assert(dim == space_train.dim);
    constexpr size_t k = 1;

    std::cerr << "[Linear Search]" << std::endl;
    auto jans =
        task::BuildAndTest<LinearIndex, space::I16>(space_train, space_test, k);

    std::cerr << "[Greedy Network]" << std::endl;
    std::cerr << "shard_count = " << shard_count << std::endl;
    std::cerr << "pool_size   = " << pool_size << std::endl;
    HierarchicalNSW search(space_train, space_train.Size());
    search.setEf(20);
    auto pans =
        task::BuildAndTest<HierarchicalNSW, space::I16>(search, space_test, k);

    auto qps = (double)1e9 / pans.search_time * space_test.Size();
    std::cerr << "QPS=" << qps << std::endl;

    size_t sum_recall = 0;
    for (size_t i = 0; i < space_test.Size(); ++i) {
        using dist_t = space::I16::dist_t;
        std::vector<dist_t> jres, pres;
        auto comp = space_train.GetComputer(space_test.GetPoint(i));
        for (size_t v : jans.ans[i])
            jres.push_back(comp.Distance(v));
        for (size_t v : pans.ans[i])
            pres.push_back(comp.Distance(v));

        std::ranges::sort(jres);
        std::ranges::sort(pres);
        for (size_t i = 0; i < k; ++i)
            if (pres[i] <= jres[i])
                ++sum_recall;
    }
    double recall = (double)sum_recall / (space_test.Size() * k) * 100;
    std::cerr << "recall=" << recall << std::endl;
    std::cout << std::fixed << std::setprecision(3) << qps << " " << recall << std::endl;
}