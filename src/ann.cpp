#include "ann.hpp"
#include <algorithm>
#include <random>
#include <cassert>
#include <cmath>

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
	return true;
}

void NeuralNetwork::save( const std::string &path )
{

}

std::vector<float> NeuralNetwork::apply( const std::vector<float> &input )
{
	// Calculate the successive layer activations
	std::vector<float> current_activation = input;
	for(const auto &layer: layers)
	{
		current_activation = apply_layer(input, layer);
	}

	// Apply the softmax function for the last output
	float exp_total = 0.0f;
	for(auto &activation: current_activation)
	{
		activation = exp(activation);
		exp_total += activation;
	}

	for(auto &activation: current_activation)
	{
		activation /= exp_total;
	}

	// Return the final output
	return current_activation;
}

void NeuralNetwork::train( const std::vector<Example> &examples )
{

}

std::vector<float> NeuralNetwork::apply_layer( const std::vector<float> &input, const NeuralNetwork::Layer &layer )
{
	std::vector<float> output(layer.neurons.size(), 0.0f);
	
	for(size_t neuron_id = 0; neuron_id < input.size(); ++neuron_id)
	{
		output[neuron_id] = neuron_activation(input, layer.function, layer.neurons[neuron_id]);
	}

	return output;
}

float NeuralNetwork::neuron_activation( 
	const std::vector<float> &input,
	NeuralNetwork::Function function,
	const NeuralNetwork::Neuron &neuron
)
{
	// Calculate the weighted sum of the input
	float result = neuron.bias;
	for(size_t i = 0; i < neuron.weights.size(); ++i)
	{
		result += input[i] * neuron.weights[i];
	}

	// Then pass it to the non-linear function of the layer
	return apply_function(result, function);
}

float NeuralNetwork::apply_function( float input, NeuralNetwork::Function function )
{
	switch( function ) {
		case NeuralNetwork::Function::RELU:
			return (input < 0) ? 0.0f: input;
		case NeuralNetwork::Function::SIGMOID:
			return 1.0f / (1.0f + expf(-input));
		default:
			return input;
	}
}