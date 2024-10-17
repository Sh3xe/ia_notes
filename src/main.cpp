#include "img_data.hpp"
#include "knn.hpp"

#include <iostream>

float test_knn(
	size_t sample_size,
	size_t threads_count,
	const std::vector<Image> &training_images,
	const std::vector<uint8_t> &training_labels,
	const std::vector<Image> &test_images,
	const std::vector<uint8_t> &test_labels
)
{
	size_t good_guess = 0;
	for( size_t i = 0; i < sample_size; ++i )
	{
		uint8_t guess = knn(test_images[i], training_images, training_labels);
		if( guess == test_labels[i] )
			++good_guess;
	}

	return (float)good_guess / (float)sample_size;
}

int main()
{
	auto training_images = load_images("../dataset/train-images.idx3-ubyte");
	auto training_labels = load_labels("../dataset/train-labels.idx1-ubyte");

	auto test_images = load_images("../dataset/t10k-images.idx3-ubyte");
	auto test_labels = load_labels("../dataset/t10k-labels.idx1-ubyte");

	size_t SAMPLE_SIZE = 50;
	std::cout << test_knn(SAMPLE_SIZE, 8, training_images, training_labels, test_images, test_labels) * 100 << "%" << std::endl;

	return 0;
}