#pragma once

#include "compute_graph.hpp"
#include "neural_network.hpp"

class Optimizer
{
public:
	Optimizer(NN *network);

	void zero_grad();

	void step();

	void accumulate(const CG::Value &value);

private:
	NN *m_network;
};