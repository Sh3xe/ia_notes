#include "ann.hpp"
#include <algorithm>
#include <random>
#include <cassert>
#include <cmath>
#include <fstream>

NeuralNetwork::NeuralNetwork(
	uint32_t input_size,
	const std::vector<LayerDescription> &layers_desc ):
	input_size(input_size)
{
	assert(layers_desc.size() >= 1 && "Neural network cannot be empty");

	construct(input_size, layers_desc, input_size);
}

bool NeuralNetwork::load( const std::string &path )
{
	std::fstream file {path, std::ios::in};

	if(!file)
	{
		return false;
	}

	this->clear();

	size_t layer_size = 0;
	file >> layer_size >> input_size;

	layers.reserve(layer_size);

	for(size_t i = 0; i < layer_size; ++i)
	{
		size_t neuron_count = 0;
		std::string func;
		file >> neuron_count >> func; 

		std::vector<NeuralNetwork::Neuron> neurons;

		neurons.reserve(neuron_count);

		for(size_t j = 0; j < neuron_count; ++j)
		{
			float bias = 0.0f, weight = 0.0f;
			size_t weight_count = 0;
			file >> weight_count >> bias;

			std::vector<float> weights;
			weights.reserve(weight_count);
			for(size_t k = 0; k < weight_count; ++k)
			{
				file >> weight;
				weights.push_back(weight);
			}

			neurons.emplace_back(weights, bias);
		}

		layers.emplace_back(function_from_name(func), neurons);
	}

	output_size = layers.rbegin()->neurons.size();

	if( file.bad() )
		return false;

	file.close();
	return true;
}

NeuralNetwork::Function NeuralNetwork::function_from_name(const std::string &name)
{
	if(name == "RELU")
		return NeuralNetwork::Function::RELU;
	else if(name == "SIGMOID")
		return NeuralNetwork::Function::SIGMOID;
	return NeuralNetwork::Function::IDENTITY;
}

void NeuralNetwork::clear()
{
	layers.clear();
	input_size = 0;
	output_size = 0;
}

bool NeuralNetwork::save( const std::string &path )
{
	std::fstream out_file {path, std::ios::out};

	if( !out_file )
	{
		return false;
	}

	// Write the data of the model in ASCII format
	out_file << layers.size() << " " << input_size << " ";

	for(const auto &layer: layers)
	{
		out_file << layer.neurons.size() << " " << function_name(layer.function) << " ";

		for(const auto &neuron: layer.neurons)
		{
			out_file << neuron.weights.size() << " " << neuron.bias << " ";
			for(float weight: neuron.weights)
			{
				out_file << weight << " ";
			}
		}
	}

	out_file.close();
	return true;
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

void NeuralNetwork::apply_and_save(
	const std::vector<float> &input,
	NeuralNetwork::BackPropagationData &data )
{
	
}

void NeuralNetwork::train( const std::vector<Example> &examples )
{

}

// TODO: redo the calculation; take softmax into account; fill the apply_and_save function; implement the save_to_disk function
NeuralNetwork::BackPropagationData NeuralNetwork::calculate_gradient(const Example &example)
{
	NeuralNetwork::BackPropagationData backpropagation_data;
	const auto &[input, expected_output] = example;

	// Step 1: apply the network to the example
	apply_and_save(input, backpropagation_data);

	// Step 2: calculate the derivate of the error over the output neurons
	auto &layer_data = backpropagation_data[layers.size()-1];
	for(size_t i = 0; i < expected_output.size(); ++i)
	{
		layer_data[i].d_err = -2*(expected_output[i] - layer_data[i].output);
	}

	// Step 3: for each layer, calculate the derivate of the error over the weights and biases and then calculate \nabla_i^(l-1) and repeate reaching the first layer
	for( size_t current_layer_id = layers.size() - 1 ; current_layer_id > 0; --current_layer_id )
	{
		auto &layer = backpropagation_data[current_layer_id];

		// For each neuron in the layer
		for(size_t i = 0; i < layer.size(); ++i)
		{
			float d0 = 
				layer[i].d_err * 
				apply_derivate(
					backpropagation_data[current_layer_id-1][i].activation,
					layers[current_layer_id].function );
				
			// Calculate dErr/dw_{ij} and dErr/db_i for the layer
			for(size_t j = 0; j < layer[i].d_weights.size(); ++j)
			{
				layer[i].d_weights[j] = d0 * backpropagation_data[current_layer_id-1][j].output;
				layer[i].d_bias = d0;
			}

			// Calculate dErr/dx_{i} for the next layer
			float derr_over_dxi = 0.0f;
			for(size_t j = 0; j < layer[i].d_weights.size(); ++j)
			{
				derr_over_dxi += d0*layers[current_layer_id].neurons[i].weights[j];
			}

			backpropagation_data[current_layer_id-1][i].d_err = derr_over_dxi;
		}
	}

	return backpropagation_data;
}

void NeuralNetwork::construct(
	uint32_t input_size,
	const std::vector<LayerDescription> &layers_desc,
	bool random )
{
	std::mt19937 random_engine;
	std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

	auto generate_function = [&]() { return random ? distribution(random_engine): 0.0f; };

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
				generate_function
			);
			neurons.emplace_back(weights, generate_function());
		}

		layers.emplace_back(layers_desc[i].function, neurons );
	}

	output_size = layers_desc.rbegin()->size;
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
		case NeuralNetwork::Function::IDENTITY:
			return input;
		default:
			return input;
	}
}

std::string NeuralNetwork::function_name( NeuralNetwork::Function function )
{
	switch( function ) {
		case NeuralNetwork::Function::RELU:
			return "RELU";
		case NeuralNetwork::Function::SIGMOID:
			return "SIGMOID";
		case NeuralNetwork::Function::IDENTITY:
			return "IDENTITY";
		default:
			return "";
	}
}

float NeuralNetwork::apply_derivate( float input, NeuralNetwork::Function function )
{
	switch( function ) {
		case NeuralNetwork::Function::RELU:
			return (input < 0) ? 0.0f: 1.0f;
		case NeuralNetwork::Function::SIGMOID:
		{
			float exp_minus_x = expf(-input);
			float denom = (1.0f + exp_minus_x);
			return exp_minus_x / (denom*denom);
		}
		case NeuralNetwork::Function::IDENTITY:
			return 1.0f;
		default:
			return 0.0f;
	}
}