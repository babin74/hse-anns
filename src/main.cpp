#include <drivers/hdf5.hpp>
#include <space/i16.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <queue>

const char* dataset_dirname = "../datasets/sift-128-euclidean.hdf5";

int main() {
    auto reader = Hdf5Reader::Open(dataset_dirname);
    auto space_test = reader.ReadI16("/test");
    auto space_train = reader.ReadI16("/train");

    const int dim = space_test.dim;
    assert(dim == space_test.dim);
    assert(dim == space_train.dim);

    const int n = space_test.Size();

    auto comp = space_train.GetComputer(space_test.GetPoint(0));
    std::priority_queue<int32_t> q;

    std::cout << "dim=" << dim << std::endl;

    for (size_t i = 0; i != n; ++i) {
        q.push(comp.Distance(i));
        if (q.size() > 10)
            q.pop();
    }

    std::vector<int> dist;
    while (!q.empty())
        dist.push_back(q.top()), q.pop();
    std::reverse(dist.begin(), dist.end());
    for (int x : dist)
        std::cout << sqrt(x) << ' ';
    std::cout << std::endl;
}