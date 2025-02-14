#pragma once

#include "compute_graph.hpp"

#include <vector>
#include <string>
#include <cstdint>
#include <initializer_list>
#include <utility>

namespace NN
{

/**
 * @brief Specification of a NeuralNetwork layer
 * 
 */
struct Layer
{
	enum class Func { SOFTMAX, LINEAR, RELU };

	Func operation;
	int input_size;
	int output_size;
};

/**
 * @brief Each element of the output layer is an affine function of the input
 * 
 * @param input_size 
 * @param output_size 
 * @return Layer 
 */
Layer linear(int input_size, int output_size);

/**
 * @brief Applies relu to the previous layer
 * 
 * @return Layer 
 */
Layer relu();

/**
 * @brief Applies softmax to the previous layer
 * 
 * @return layer
 */
Layer softmax();

class NeuralNet
{
public:
	NeuralNet( const std::initializer_list<Layer> &layer_desc );

	std::vector<CG::Value> forward(const std::vector<double> &input);

	bool save_weights(const std::string &path);
	
	bool load_weights(const std::string &path);
	
	friend class Optimizer;
private:
	std::pair<std::vector<CG::Value>, std::vector<CG::Value>> construct_tree(bool random = true);

	std::vector<Layer> m_architecture;
	std::vector<CG::Value> m_output_weights;
	std::vector<CG::Value> m_input_weights;
};

} // namespace NN