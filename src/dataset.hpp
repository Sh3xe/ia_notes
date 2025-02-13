#pragma once

#include <vector>
#include <utility>
#include <cstdint>

std::pair<std::vector<std::vector<double>>, std::vector<uint32_t>> load_mnist_digits_train();

std::pair<std::vector<std::vector<double>>, std::vector<uint32_t>> load_mnist_digits_test();