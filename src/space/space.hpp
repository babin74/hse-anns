#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace space {

namespace impl {

template <typename T>
concept Distance = std::integral<T> || std::floating_point<T>;

template <typename S, typename C>
concept ConceptSpaceTypes = requires {
    typename S::dist_t;
    typename C::dist_t;
    typename S::point_t;
    typename S::computer_t;
    std::is_same_v<typename S::dist_t, typename C::dist_t> == true;
}
&&Distance<typename S::dist_t>;

template <typename S, typename C>
concept ConceptSpaceMethods =
    requires(S space, const S& const_space, const C& comp,
             typename S::point_t q, size_t idx) {
    { space.GetComputer(idx) } -> std::same_as<C>;
    { space.GetComputer(q) } -> std::same_as<C>;
    { const_space.Size() } -> std::same_as<size_t>;
    { comp.Distance(idx) } -> std::same_as<typename C::dist_t>;
};

template <typename S, typename C>
concept CheckSpace = ConceptSpaceTypes<S, C> && ConceptSpaceMethods<S, C>;

}  // namespace impl

template <typename T>
concept Space = impl::CheckSpace<T, typename T::computer_t>;

}  // namespace space