#include "dataset.hpp"
#include "img_data.hpp"
#include "utils.hpp"
#include <string>

static std::vector<std::vector<double>> fetch_mnist_images(const std::string &path)
{
	auto images = load_images( path );
	std::vector<std::vector<double>> converted_images;

	for(const auto &img: images)
	{
		converted_images.push_back(img.convert_to_01_vector());
	}

	return converted_images;
}

static std::vector<uint32_t> fetch_mnist_labels(const std::string &path)
{
	auto labels = load_labels( path );
	std::vector<uint32_t> converted_labels;

	for(const auto &label: labels)
	{
		converted_labels.push_back(static_cast<uint32_t>(label));
	}

	return converted_labels;
}

std::pair<std::vector<std::vector<double>>, std::vector<uint32_t>> load_mnist_digits_train()
{
	return std::make_pair(
		fetch_mnist_images("../dataset/train-images.idx3-ubyte"),
		fetch_mnist_labels("../dataset/train-labels.idx1-ubyte")
	);
}

std::pair<std::vector<std::vector<double>>, std::vector<uint32_t>> load_mnist_digits_test()
{
	return std::make_pair(
		fetch_mnist_images("../dataset/t10k-images.idx3-ubyte"),
		fetch_mnist_labels("../dataset/t10k-labels.idx1-ubyte")
	);
}