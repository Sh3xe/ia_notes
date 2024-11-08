#include "img_data.hpp"
#include "knn.hpp"
#include "ann.hpp"

#include "utils.hpp"
#include <iostream>

float test_knn(
	size_t sample_size,
	size_t threads_count,
	const std::vector<Image> &training_images,
	const std::vector<uint8_t> &training_labels,
	const std::vector<Image> &test_images,
	const std::vector<uint8_t> &test_labels )
{
	size_t good_guess = 0;
	auto permutation = generate_permutation(test_images.size());

	for( size_t i = 0; i < sample_size; ++i )
	{
		uint8_t guess = knn(test_images[permutation[i]], training_images, training_labels);
		if( guess == test_labels[i] )
			++good_guess;
	}

	return (float)good_guess / (float)sample_size;
}

float test_ann(
	size_t sample_size,
	size_t threads_count,
	const std::vector<Image> &training_images,
	const std::vector<uint8_t> &training_labels,
	const std::vector<Image> &test_images,
	const std::vector<uint8_t> &test_labels )
{
	NeuralNetwork network(27*27, 10, {
		NeuralNetwork::LayerDescription(NeuralNetwork::Function::RELU, 16),
		NeuralNetwork::LayerDescription(NeuralNetwork::Function::RELU, 16)
	});

	network.load("../models/numbers.txt");

	size_t good_guess = 0;
	for( size_t i = 0; i < sample_size; ++i )
	{
		auto guess = network.apply(test_images[i].convert_to_01_vector());
		uint32_t response = 0;
		for(uint32_t j = 0; j < guess.size(); ++j)
		{
			if(guess[response] < guess[j])
				response = j;	
		}

		good_guess += (response == test_labels[i]);
	}

	return (float)good_guess / (float)sample_size;
}

void train_network(
	const std::vector<Image> &training_images,
	const std::vector<uint8_t> &training_labels )
{
	// Creates a neural network
	NeuralNetwork network(27*27, 10, {
		NeuralNetwork::LayerDescription(NeuralNetwork::Function::RELU, 16),
		NeuralNetwork::LayerDescription(NeuralNetwork::Function::RELU, 16)
	});
	
	// Converts the training data to float vectors
	std::vector<NeuralNetwork::Example> examples;
	for(size_t i = 0; i < training_images.size(); ++i)
	{
		std::vector<float> answer(10, 0.0f);
		answer[(size_t)training_labels[i]] = 1.0f;
		examples.emplace_back(std::make_pair(training_images[i].convert_to_01_vector(), answer));
	}

	// Train the neural network
	network.train(examples, 4, 100);

	// Saves the current weights and biases to disk
	network.save("../models/numbers.txt");
}

int main()
{
	// Loads the training and image data
	auto training_images = load_images("../dataset/train-images.idx3-ubyte");
	auto training_labels = load_labels("../dataset/train-labels.idx1-ubyte");

	auto test_images = load_images("../dataset/t10k-images.idx3-ubyte");
	auto test_labels = load_labels("../dataset/t10k-labels.idx1-ubyte");

	train_network(training_images, training_labels);
	return 0;
}