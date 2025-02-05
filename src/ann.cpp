#include "ann.hpp"
#include <algorithm>
#include <random>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include "utils.hpp"

NeuralNetwork::NeuralNetwork(
	uint32_t input_size,
	uint32_t output_size,
	const std::vector<LayerDescription> &layers_desc ):
	input_size(input_size),
	output_size(output_size)
{
	assert(layers_desc.size() >= 1 && "Neural network cannot be empty");

	construct(input_size, output_size, layers_desc, true);
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

		for(auto &act: current_activation)
			act = apply_function(act, layer.function);
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

NeuralNetwork::BackPropagationData NeuralNetwork::construct_empty_backprop_data()
{
	NeuralNetwork::BackPropagationData data;
	data.reserve(layers.size());

	for(size_t l = 0; l < layers.size(); ++l)
	{
		uint32_t previous_size = (l == 0)? input_size: layers[l-1].neurons.size();

		NeuralNetwork::BackPropagationLayerData layer_data;
		layer_data.reserve(layers[l].neurons.size());

		// Initialises each neuron with random weights and bias
		for(uint32_t i = 0; i < layers[l].neurons.size(); ++i)
		{
			NeuralNetwork::BackPropagationNeuronData neuron_data(previous_size);
			layer_data.push_back(neuron_data);
		}

		data.push_back(layer_data);
	}

	return std::move(data);
}

std::vector<float> NeuralNetwork::apply_and_save(
	const std::vector<float> &input,
	NeuralNetwork::BackPropagationData &data )
{
	// Calculate the successive layer activations
	std::vector<float> current_activation = input;

	for(size_t layer_id = 0; layer_id < layers.size(); ++layer_id)
	{
		// Apply the layer jute like the "apply function"
		current_activation = apply_layer(input, layers[layer_id]);

		// saves it in "data"
		for(size_t i = 0; i < current_activation.size(); ++i)
		{
			data[layer_id][i].activation = current_activation[i];
			current_activation[i] = apply_function(current_activation[i], layers[layer_id].function);
			data[layer_id][i].output = current_activation[i];
		}
	}

	// Apply the softmax function for the last output
	assert(current_activation.size() == data[layers.size()-1].size());
	float exp_total = 0.0f;
	for(size_t i = 0; i < current_activation.size(); ++i)
	{
		auto &neuron = data[layers.size()-1][i];
		neuron.output = exp(neuron.output);
		exp_total += neuron.output;
	}

	for(size_t i = 0; i < current_activation.size(); ++i)
	{
		data[layers.size()-1][i].output /= exp_total;
	}

	return current_activation;
}

void NeuralNetwork::train( const std::vector<Example> &examples, size_t thread_count, size_t max_iteration )
{
	const float constant = 1e-2 / (float)examples.size();

	// Randomize the examples
	auto permutation = generate_permutation(examples.size());
	size_t current_example = 0;

	// Until a max iteration range
	for(size_t i = 0; i < max_iteration; ++i)
	{
		// Calculate the mean of the gradient over a batch examples
		auto mean_backprop = construct_empty_backprop_data();
		
		for(size_t k = 0; k < 300; ++k)
		{
			// Gradient for a single example
			auto backprop_data = calculate_gradient(examples[current_example]);
			current_example = (current_example+1) % examples.size();

			// Copy into "mean_backprop"
			for(size_t bpl = 0; bpl < backprop_data.size(); ++bpl)
			{
				for(size_t bpn = 0; bpn < backprop_data[bpl].size(); ++bpn)
				{
					auto &mean_n = mean_backprop[bpl][bpn];
					auto &n = backprop_data[bpl][bpn];
					mean_n.d_bias += n.d_bias;
					for(size_t w = 0; w < mean_n.d_weights.size(); ++w)
					{
						mean_n.d_weights[w] += n.d_weights[w];
					}
				}
			}
		}

		// Gradient descent with the delta rule using the constant defined at the beginning of the function
		for(size_t bpl = 0; bpl < mean_backprop.size(); ++bpl)
		{
			for(size_t bpn = 0; bpn < mean_backprop[bpl].size(); ++bpn)
			{
				auto &mean_n = mean_backprop[bpl][bpn];
				auto &n = layers[bpl].neurons[bpn];
				n.bias += constant * mean_n.d_bias;

				for(size_t w = 0; w < mean_n.d_weights.size(); ++w)
				{
					n.weights[w] += constant * mean_n.d_weights[w];
				}
			}
		}
	}
}

NeuralNetwork::BackPropagationData NeuralNetwork::calculate_gradient(const Example &example)
{
	auto backpropagation_data = construct_empty_backprop_data();
	const auto &[input, expected_output] = example;

	// Step 1: apply the network to the example
	auto output = apply_and_save(input, backpropagation_data);

	// Step 2: calculate the derivate of the error over the output neurons
	auto &layer_data = backpropagation_data[layers.size()-1];
	for(size_t i = 0; i < expected_output.size(); ++i)
	{
		float d_err = 0.0f;
		for(size_t k = 0; k < expected_output.size(); ++k)
		{
			if(fabs(expected_output[k]) < 0.01f)
				continue;
			
			if(fabs(layer_data[k].output) < 0.01f)
				layer_data[k].output = 0.01f;
			
			d_err += output[i]*output[k];
			if(i == k)
			{
				d_err -= (expected_output[k] / layer_data[k].output) * output[i];
			}
		}

		layer_data[i].d_err = d_err;
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
	uint32_t output_size,
	const std::vector<LayerDescription> &layers_desc,
	bool random )
{
	std::mt19937 random_engine;
	std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

	auto generate_function = [&]() { return random ? distribution(random_engine): 0.0f; };

	auto full_layer_desc = layers_desc;
	full_layer_desc.emplace_back(LayerDescription(Function::IDENTITY, output_size));

	for(size_t i = 0; i < full_layer_desc.size(); ++i)
	{
		uint32_t previous_size = (i == 0)? input_size: full_layer_desc[i-1].size;
		std::vector<Neuron> neurons;
		neurons.reserve(full_layer_desc[i].size);

		// Initialises each neuron with random weights and bias
		for(uint32_t j = 0; j < full_layer_desc[i].size; ++j)
		{
			std::vector<float> weights(previous_size, 0.0f);
			std::generate_n(
				weights.begin(), 
				previous_size, 
				generate_function
			);
			neurons.emplace_back(weights, generate_function());
		}

		layers.emplace_back(full_layer_desc[i].function, neurons );
	}

	output_size = full_layer_desc.rbegin()->size;
}

std::vector<float> NeuralNetwork::apply_layer( const std::vector<float> &input, const NeuralNetwork::Layer &layer )
{
	std::vector<float> output(layer.neurons.size(), 0.0f);

	for(size_t neuron_id = 0; neuron_id < layer.neurons.size(); ++neuron_id)
	{
		output[neuron_id] = neuron_activation(input, layer.neurons[neuron_id]);
	}

	return output;
}

float NeuralNetwork::neuron_activation( 
	const std::vector<float> &input,
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
	return result;
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