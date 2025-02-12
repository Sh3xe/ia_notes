#pragma once

#include "compute_graph.hpp"

#include <vector>
#include <string>
#include <cstdint>
#include <initializer_list>
#include <utility>

namespace NN
{

struct Layer
{
	enum class Func { SOFTMAX, LINEAR, RELU };

	Func operation;
	int input_size;
	int output_size;
};

Layer linear(int input_size, int output_size);

Layer relu();

Layer softmax();

class NeuralNet
{
public:
	NeuralNet( const std::initializer_list<Layer> &layer_desc );

	std::vector<CG::Value> forward(const std::vector<double> &input);

	// bool save_weights(const std::string &path);
	
	// bool load_weights(const std::string &path);
	
	friend class Optimizer;
private:
	std::pair<std::vector<CG::Value>, std::vector<CG::Value>> construct_tree(bool random = true);

	std::vector<Layer> m_architecture;
	std::vector<CG::Value> m_output_weights;
	std::vector<CG::Value> m_input_weights;
};

} // namespace NN