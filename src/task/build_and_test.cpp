#include <chrono>
#include <drivers/hdf5.hpp>
#include <index/greedy-net.hpp>
#include <index/linear.hpp>
#include <iomanip>
#include <ios>
#include <numeric>
#include <random>
#include <string_view>
#include <task/build_and_test.hpp>
#include "index/hnsw.hpp"

int task::TestGreedyNetParams(int argc, char** argv) {
    std::vector<size_t> shard_v(16);
    std::iota(shard_v.begin(), shard_v.end(), size_t(1));
    std::vector<size_t> pools_v(60);
    std::iota(pools_v.begin(), pools_v.end(), size_t(0));

    if (argc <= 3) {
        std::cerr << "Usage: tann test greedy-net <dataset>";
        return 1;
    }
    std::string_view dataset_dirname = argv[3];
    auto reader = Hdf5Reader::Open(dataset_dirname);
    auto space_test = reader.ReadI16("/test");
    auto space_train = reader.ReadI16("/train");

    const size_t dim = space_test.dim;
    if (dim != space_train.dim) {
        std::cerr << "Incorrect dataset" << std::endl;
        return 1;
    }

    std::mt19937 rng(1337228);
    const size_t n = space_train.Size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = rng() % (i + 1);
        if (i != j) {
            for (size_t u = 0; u < dim; ++u) {
                // std::cout << dim * i + u << " " << dim * j + u << ", sz=" << space_train.data.size() << std::endl;
                std::swap(space_train.data[dim * i + u],
                          space_train.data[dim * j + u]);
            }
        }
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
        double avg_dist_eval;
    };
    std::vector<RunInfo> runs;

    for (size_t shard_count : shard_v) {
        using hc = std::chrono::high_resolution_clock;
        std::cerr << "[Greedy Net, shard_count=" << shard_count << "]"
                  << std::flush;

        auto ts_build_start = hc::now();
        GreedyNet searcher(space_train);
        searcher.shard_count = shard_count;
        searcher.Build();
        std::cerr << " => build time: " << (hc::now() - ts_build_start).count()
                  << " ns" << std::endl;

        for (size_t pool_size : pools_v) {
            searcher.pool_size = pool_size;
            searcher.eval_dist = 0;
            auto pans = JustRun(searcher, space_test, k, false);

            auto qps = (double)1e9 / pans.search_time * space_test.Size();

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
            double avg_dist_eval =
                (double)searcher.eval_dist / (space_test.Size());
            // std::cerr << "pool_size=" << pool_size << ", ";
            // std::cerr << "QPS=" << qps << ", ";
            // std::cerr << "recall=" << recall << std::endl;
            runs.push_back(
                RunInfo{shard_count, pool_size, qps, recall, avg_dist_eval});
        }

        // std::cerr << std::endl;
    }

    // Print json info
    std::cout << "[";
    std::cout << std::fixed << std::setprecision(4);
    bool first = true;
    for (auto [s, p, q, r, k] : runs) {
        if (!first)
            std::cout << ",";
        else
            first = false;
        std::cout << "[";
        std::cout << s << "," << p << "," << q << "," << r << "," << k;
        std::cout << "]";
    }
    std::cout << "]" << std::endl;

    return 0;
}

int task::TestHnswParams(int argc, char** argv) {
    std::vector<size_t> efconstruct_v{{5, 10, 20, 50, 100, 200}};
    std::vector<size_t> m_v{{10, 20, 40, 50}};
    std::vector<size_t> ef_v{{5, 10, 20, 30}};
    if (argc <= 3) {
        std::cerr << "Usage: tann test hnsw <dataset>";
        return 1;
    }
    std::string_view dataset_dirname = argv[3];
    auto reader = Hdf5Reader::Open(dataset_dirname);
    auto space_test = reader.ReadI16("/test");
    auto space_train = reader.ReadI16("/train");

    const size_t dim = space_test.dim;
    if (dim != space_train.dim) {
        std::cerr << "Incorrect dataset" << std::endl;
        return 1;
    }

    std::mt19937 rng(1337228);
    const size_t n = space_train.Size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = rng() % (i + 1);
        if (i != j) {
            for (size_t u = 0; u < dim; ++u) {
                // std::cout << dim * i + u << " " << dim * j + u << ", sz=" << space_train.data.size() << std::endl;
                std::swap(space_train.data[dim * i + u],
                          space_train.data[dim * j + u]);
            }
        }
    }

    constexpr size_t k = 1;

    std::cerr << "[Linear Search]" << std::endl;
    auto jans =
        task::BuildAndTest<LinearIndex, space::I16>(space_train, space_test, k);

    struct RunInfo {
        size_t efconstruct;
        size_t m;
        size_t ef;
        double qps;
        double recall;
        double avg_dist_eval;
    };
    std::vector<RunInfo> runs;

    for (size_t efconstruct : efconstruct_v) {
        for (size_t M : m_v) {
            using hc = std::chrono::high_resolution_clock;
            std::cerr << "[HSNW, efconstruct=" << efconstruct << ", M=" << M
                      << "]" << std::flush;

            auto ts_build_start = hc::now();
            HierarchicalNSW searcher(space_train, space_train.Size(), M,
                                     efconstruct);
            searcher.Build();
            std::cerr << " => build time: "
                      << (hc::now() - ts_build_start).count() << " ns"
                      << std::endl;

            for (size_t ef : ef_v) {
                searcher.metric_distance_computations = 0;
                searcher.setEf(ef);
                auto pans = JustRun(searcher, space_test, k, false);

                auto qps = (double)1e9 / pans.search_time * space_test.Size();

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
                double recall =
                    (double)sum_recall / (space_test.Size() * k) * 100;
                double avg_dist_eval =
                    (double)searcher.metric_distance_computations /
                    (space_test.Size());
                // std::cerr << "pool_size=" << pool_size << ", ";
                // std::cerr << "QPS=" << qps << ", ";
                // std::cerr << "recall=" << recall << std::endl;
                runs.push_back(
                    RunInfo{efconstruct, M, ef, qps, recall, avg_dist_eval});
            }

            // std::cerr << std::endl;
        }
    }

    // Print json info
    std::cout << "[";
    std::cout << std::fixed << std::setprecision(4);
    bool first = true;
    for (auto [efc, m, ef, q, r, k] : runs) {
        if (!first)
            std::cout << ",";
        else
            first = false;
        std::cout << "[";
        std::cout << efc << "," << m << "," << ef << "," << q << "," << r << "," << k;
        std::cout << "]";
    }
    std::cout << "]" << std::endl;

    return 0;
}