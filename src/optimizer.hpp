#pragma once

#include "compute_graph.hpp"
#include "neural_network.hpp"

namespace NN
{

/**
 * @brief Implements the gradient descent for a given NeuralNet
 * 
 */
class Optimizer
{
public:

	/**
	 * @brief Construct a new Optimizer object
	 * 
	 * @param network The network which weights we want to optimize
	 * @param learning_rate 
	 * @param momentum The fraction of the previous gradient that will be added together with the new grad in Optimizer::step
	 */
	Optimizer(
		const NeuralNet &network, 
		double learning_rate=0.001,
		double momentum = 0.9
	);

	/**
	 * @brief Reset the stored gradient
	 * 
	 */
	void zero_grad();

	/**
	 * @brief Applies one gradient descent step with the accumulated gradient
	 * 
	 * @return * void 
	 */
	void step();

	/**
	 * @brief 
	 * 
	 * @param value The output of a loss function (ex, CG::cross_entropy)
	 */
	void accumulate(const CG::Value &value);

	/**
	 * @brief L2 norm of the accumulated gradient, that is sqrt(sum_i w_i^2) where w_i are the differential's weights
	 * 
	 * @return double 
	 */
	double grad_l2_norm();

private:
	// The network's output weights
	std::vector<CG::Value> m_network_weights;

	// Parameters
	double m_learning_rate = 0.0;
	double m_momentum = 0.0;

	// How many gradient where added (used for normalizing the gradient)
	size_t m_accumulated_count = 0;
};

};