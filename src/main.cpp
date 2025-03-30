#include <chrono>
#include <drivers/hdf5.hpp>
#include <iomanip>
#include <limits>
#include <space/i16.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <queue>
#include <vector>
#include "index/greedy-net.hpp"
#include "index/linear.hpp"
#include "task/build_and_test.hpp"

const char* dataset_dirname = "../datasets/siftsmall-128-euclidean.hdf5";

int main(int argc, char** argv) {
    assert(argc == 2);
    const int shard_count = atoi(argv[1]);

    auto reader = Hdf5Reader::Open(dataset_dirname);
    auto space_test = reader.ReadI16("/test");
    auto space_train = reader.ReadI16("/train");

    const int dim = space_test.dim;
    assert(dim == space_test.dim);
    assert(dim == space_train.dim);
    constexpr size_t k = 1;

    // std::cerr << "[Linear Search]" << std::endl;
    auto jans =
        task::BuildAndTest<LinearIndex, space::I16>(space_train, space_test, k, false);

    // std::cerr << "[Greedy Network]" << std::endl;
    GreedyNet search_greedy(space_train);
    search_greedy.shard_count = shard_count;
    auto pans =
        task::BuildAndTest<GreedyNet, space::I16>(search_greedy, space_test, k, false);

    auto qps = (double)1e9 / pans.search_time * space_test.Size();
    // std::cerr << "QPS=" << qps << std::endl;

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
    // std::cerr << "recall=" << recall << std::endl;
    std::cout << std::fixed << std::setprecision(3) << qps << " " << recall << std::endl;
}