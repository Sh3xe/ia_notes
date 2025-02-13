#pragma once

#include "compute_graph.hpp"
#include "neural_network.hpp"

namespace NN
{

class Optimizer
{
public:
	Optimizer(
		const NeuralNet &network, 
		double learning_rate=0.001,
		double momentum = 0.9
	);

	void zero_grad();

	void step();

	void accumulate(const CG::Value &value);

private:
	std::vector<CG::Value> m_network_weights;
	double m_learning_rate = 0.0;
	double m_momentum = 0.0;
};

};