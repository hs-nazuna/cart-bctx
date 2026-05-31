#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <algorithm>
#include <functional>
#include <set>

#ifdef CART_BCTX_NAMESPACE
#define BEGIN_NAMESPACE namespace CART_BCTX_NAMESPACE {
#define END_NAMESPACE }
#define USE_NAMESPACE using namespace CART_BCTX_NAMESPACE;
#else
#define BEGIN_NAMESPACE
#define END_NAMESPACE
#define USE_NAMESPACE
#endif

BEGIN_NAMESPACE

template<typename T> using Vector = std::vector<T>;
template<typename T> using Matrix = std::vector<Vector<T>>;
template<typename T> T square(T x) { return x * x; }
template<typename T> int argmax(const Vector<T>& vec) {
    return static_cast<int>(std::distance(vec.begin(), std::max_element(vec.begin(), vec.end())));
}

END_NAMESPACE