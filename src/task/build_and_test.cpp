#include <chrono>
#include <drivers/hdf5.hpp>
#include <index/greedy-net.hpp>
#include <index/linear.hpp>
#include <iomanip>
#include <ios>
#include <numeric>
#include <string_view>
#include <task/build_and_test.hpp>

int task::TestGreedyNetParams(int argc, char** argv) {
    std::vector<size_t> shard_v = {1, 2, 4, 8, 16};
    std::vector<size_t> pools_v(30);
    
    // std::vector<size_t> shard_v = {4};
    // std::vector<size_t> pools_v(30);
    std::iota(pools_v.begin(), pools_v.end(), size_t(0));

    if (argc <= 3) {
        std::cerr << "Usage: tann test greedy-net <dataset>";
        return 1;
    }
    std::string_view dataset_dirname = argv[3];
    auto reader = Hdf5Reader::Open(dataset_dirname);
    auto space_test = reader.ReadI16("/test");
    auto space_train = reader.ReadI16("/train");


    const int dim = space_test.dim;
    if (dim != space_train.dim) {
        std::cerr << "Incorrect dataset" << std::endl;
        return 1;
    }
    
    constexpr size_t k = 1;

    std::cerr << "[Linear Search]" << std::endl;
    auto jans =
        task::BuildAndTest<LinearIndex, space::I16>(space_train, space_test, k);

    struct RunInfo {
        size_t shard_count;
        size_t pool_size;
        double qps;
        double recall;
    };
    std::vector<RunInfo> runs;

    for (size_t shard_count : shard_v) {
        using hc = std::chrono::high_resolution_clock;
        std::cerr << "[Greedy Net, shard_count=" << shard_count << "]"
                  << std::endl;

        auto ts_build_start = hc::now();
        GreedyNet searcher(space_train);
        searcher.shard_count = shard_count;
        searcher.Build();
        std::cerr << "Build time: " << (hc::now() - ts_build_start).count()
                  << " ns" << std::endl;

        for (size_t pool_size : pools_v) {
            std::cerr << "Set pool_size=" << pool_size << std::endl;
            searcher.pool_size = pool_size;
            auto pans = JustRun(searcher, space_test, k);

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

            runs.push_back(RunInfo{ shard_count, pool_size, qps, recall });
        }

        std::cerr << std::endl;
    }

    // Print json info
    std::cout << "[";
    std::cout << std::fixed << std::setprecision(4);
    bool first = true;
    for (auto [s, p, q, r] : runs) {
        if (!first) std::cout << ",";
        else first = false;
        std::cout << "[";
        std::cout << s << "," << p << "," << q << "," << r;
        std::cout << "]";
    }
    std::cout << "]" << std::endl;

    return 0;
}