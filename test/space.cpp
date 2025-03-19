#include <catch2/catch_test_macros.hpp>

#include <optional>
#include <space/i16.hpp>
#include <space/space.hpp>

template <space::Space S>
std::pair<size_t, size_t> FindNearestPoints(S& space) {
    using dist_t = S::dist_t;

    const size_t n = space.Size();
    std::optional<dist_t> min_dist;
    std::pair<size_t, size_t> res;

    for (size_t i = 0; i < n; ++i) {
        auto comp = space.GetComputer(i);
        for (size_t j = i + 1; j < n; ++j) {
            auto dist = comp.Distance(j);
            if (!min_dist.has_value() || min_dist.value() > dist) {
                min_dist = dist;
                res = {i, j};
            }
        }
    }

    assert(min_dist.has_value());
    return res;
}

template <space::Space S>
auto Distance(S space, size_t i, size_t j) {
    auto comp = space.GetComputer(i);
    return comp.Distance(j);
}

TEST_CASE("I16 Basic") {
    space::I16 space;
    space.dim = 4;
    space.data = {
        0, 0, 0, 0, // 0
        1, 0, 2, 3, // 1
        1, 0, 2, 2 // 2
    };

    REQUIRE(space.Size() == 3);

    REQUIRE(Distance(space, 0, 1) == 1+0+4+9);
    REQUIRE(Distance(space, 0, 2) == 1+0+4+4);
    REQUIRE(Distance(space, 1, 2) == 0+0+0+1);

    auto [a, b] = FindNearestPoints(space);
    REQUIRE(a == 1);
    REQUIRE(b == 2);
}