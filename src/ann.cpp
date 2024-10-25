#include "ann.hpp"
#include <algorithm>
#include <random>
#include <cassert>

NeuralNetwork::NeuralNetwork(
	uint32_t input_size,
	const std::vector<LayerDescription> &layers_desc ):
	input_size(input_size)
{
	assert(layers_desc.size() >= 1 && "Neural network cannot be empty");

	std::mt19937 random_engine;
	std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

	// Initialises the layers with null weights and biases
	for(size_t i = 0; i < layers_desc.size(); ++i)
	{
		uint32_t previous_size = (i == 0)? input_size: layers_desc[i-1].size;
		std::vector<Neuron> neurons;
		neurons.reserve(layers_desc[i].size);

		// Initialises each neuron with random weights and bias
		for(uint32_t j = 0; j < layers_desc[i].size; ++j)
		{
			std::vector<float> weights(previous_size, 0.0f);
			std::generate_n(
				weights.begin(), 
				previous_size, 
				[&]() { return distribution(random_engine);}
			);
			neurons.emplace_back(weights, distribution(random_engine));
		}

		layers.emplace_back(layers_desc[i].function, neurons );
	}

	output_size = layers_desc.rbegin()->size;
}

bool NeuralNetwork::load( const std::string &path )
{

}

void NeuralNetwork::save( const std::string &path )
{

}

std::vector<float> NeuralNetwork::apply( const std::vector<float> &input )
{

}

void NeuralNetwork::train( const std::vector<Example> &examples )
{

}