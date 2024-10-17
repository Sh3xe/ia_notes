#pragma once
#include "img_data.hpp"

uint8_t knn(
	const Image &image,
	const std::vector<Image> &training_images,
	const std::vector<uint8_t> &training_labels
);