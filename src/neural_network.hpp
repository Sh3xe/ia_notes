#pragma once

#include "compute_graph.hpp"

#include <vector>
#include <string>
#include <cstdint>
#include <initializer_list>

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

private:
	std::vector<Layer> m_architecture;
	
};