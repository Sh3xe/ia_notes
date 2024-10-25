#pragma once
#include <vector>
#include <string>
#include <cstdint>

enum class NeuralNetworkFunction 
{
	SIGMOID,
	RELU
};

struct LayerDescription
{
	LayerDescription(NeuralNetworkFunction function, uint32_t size):
		function(function),
		size(size) {}
	NeuralNetworkFunction function;
	uint32_t size;
};

/**
 * @brief Represents an artificial neural networks
 * 
 */
class NeuralNetwork
{
public:
	using Example = std::pair<std::vector<float>, std::vector<float>>;

	/**
	 * @brief Construct a new Neural Network object
	 * 
	 * @param input_size the size of an input to the network
	 * @param output_size the size of the output
	 * @param layers a description of the hidden layers
	 */
	NeuralNetwork(uint32_t input_size, const std::vector<LayerDescription> &layers);

	/**
	 * @brief Loads the current weights and biases from disk
	 * 
	 * @param path
	 * @return true in case of success
	 * @return false in case of disk failure
	 */
	bool load(const std::string &path);

	/**
	 * @brief Saves the current weights and biases to disk
	 * 
	 * @param path 
	 */
	void save(const std::string &path);

	/**
	 * @brief Apply the neural network to an input vector
	 * 
	 * @param input the input vector, must be the same size as the input of the network
	 * @return std::vector<float> the output of the neural network
	 */
	std::vector<float> apply( const std::vector<float> &input );

	/**
	 * @brief Given a list of example, this function apply a backpropagation algorithm to train the network
	 * 
	 * @param examples 
	 */
	void train( const std::vector<Example> &examples );

private:
	struct Neuron
	{
		Neuron(const std::vector<float> &weights, float bias):
			weights(weights), bias(bias) {}
		std::vector<float> weights;
		float bias;
	};

	struct NeuralNetworkLayer 
	{
		NeuralNetworkLayer(NeuralNetworkFunction function, const std::vector<Neuron> &neurons):
			function(function), neurons(neurons) {}
		NeuralNetworkFunction function;
		std::vector<Neuron> neurons;
	};

	std::vector<NeuralNetworkLayer> layers;
	uint32_t input_size;
	uint32_t output_size;
};
