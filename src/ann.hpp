#pragma once
#include <vector>
#include <string>
#include <cstdint>

/**
 * @brief Represents an artificial neural networks
 * 
 */
class NeuralNetwork
{
public:
	using Example = std::pair<std::vector<float>, std::vector<float>>;

	enum class Function 
	{
		SIGMOID,
		RELU,
		IDENTITY
	};

	struct LayerDescription
	{
		LayerDescription(NeuralNetwork::Function function, uint32_t size):
			function(function),
			size(size) {}
		NeuralNetwork::Function function;
		uint32_t size;
	};

	/**
	 * @brief Construct a new Neural Network object
	 * 
	 * @param input_size the size of an input to the network
	 * @param output_size the size of the output
	 * @param layers a description of the hidden layers
	 */
	NeuralNetwork(uint32_t input_size, uint32_t output_size, const std::vector<LayerDescription> &hidden_layers);

	/**
	 * @brief Loads the current weights and biases from disk
	 * 
	 * @param path
	 * @return true in case of success
	 * @return false in case of disk failure
	 */
	bool load(const std::string &path);

	/**
	 * @brief Serealize the current weights and biases to disk
	 * 
	 * @param path 
	 */
	bool save(const std::string &path);

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
	void train( const std::vector<Example> &examples, size_t thread_count, float max_time_seconds  = 100.0f);

private:
	struct Neuron
	{
		Neuron() {}
		Neuron(const std::vector<float> &weights, float bias):
			weights(weights), bias(bias) {}
		std::vector<float> weights;
		float bias;
	};

	struct Layer 
	{
		Layer(NeuralNetwork::Function function, const std::vector<Neuron> &neurons):
			function(function), neurons(neurons) {}
		NeuralNetwork::Function function;
		std::vector<Neuron> neurons;
	};

	/**
	 * @brief Stores the partial derivate of the error over the neuron data for a single example
	 * 
	 */
	struct BackPropagationNeuronData
	{
		BackPropagationNeuronData() {}

		BackPropagationNeuronData(size_t size):
			d_weights(size, 0.0f) {}
		
		// neuron activation
		float activation = 0.0f;

		// neuron output = f^l(activation)
		float output = 0.0f;

		// derivate of the error over the neuron output
		float d_err = 0.0f;

		// derivate of the error over the neuron bias
		float d_bias = 0.0f;

		// derivate of the error over the neuron weigths
		std::vector<float> d_weights;
	};

	/**
	 * @brief Stores all the partial derivates of the error over the network parameters as well as intermediary values for a single example
	 * 
	 */
	using BackPropagationLayerData = std::vector<BackPropagationNeuronData>;

	/**
	 * @brief Struct containing all the gradient of the error for a single example
	 * 
	 */
	using BackPropagationData = std::vector<BackPropagationLayerData>;

	BackPropagationData construct_empty_backprop_data();

	BackPropagationData calculate_gradient(const Example &example);

	std::vector<float> apply_and_save(const std::vector<float> &input, BackPropagationData &data);

	void construct( uint32_t input_size, uint32_t output_size, const std::vector<LayerDescription> &layers, bool random = true );

	std::vector<float> apply_layer( const std::vector<float> &input, const NeuralNetwork::Layer &layer );

	float neuron_activation( const std::vector<float> &input, const Neuron &neuron);

	float apply_function( float input, NeuralNetwork::Function function );

	float apply_derivate( float input, NeuralNetwork::Function function );

	/**
	 * @brief Converts the NeuralNetwork::Function union to a string
	 * 
	 * @param function 
	 * @return std::string 
	 */
	std::string function_name( NeuralNetwork::Function function );

	/**
	 * @brief inverse of "function_name"
	 * 
	 * @param name 
	 * @return NeuralNetwork::Function 
	 */
	NeuralNetwork::Function function_from_name( const std::string &name );

	/**
	 * @brief Clear all of the memory associated with the ann
	 * 
	 */
	void clear();

	// Layers of the network, the last one is by definition the output and a softmax will be applied when running the network for an input
	std::vector<Layer> layers;

	// size of the input
	uint32_t input_size;

	// size of the output
	uint32_t output_size;
};
