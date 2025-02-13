#include "optimizer.hpp"
#include "utils.hpp"

#include <cassert>
#include <cmath>

namespace NN
{

Optimizer::Optimizer(
		const NeuralNet &net, 
		double learning_rate,
		double momentum
	)
	: m_learning_rate(learning_rate),
	m_momentum(momentum)
{
	// Topological is deterministic and for two CG::Value with the same graph,
	// it will yield the same order. This is how we can pair every weights from
	// the neural net, with it's loss counterpart
	m_network_weights = topological_sort(net.m_output_weights);
}

void Optimizer::zero_grad()
{
	for(const auto &v: m_network_weights)
		v->m_diff = 0.0;
}

double Optimizer::grad_l2_norm()
{
	double res = 0.0;
	for(const auto &v: m_network_weights)
	{
		res += v->m_diff*v->m_diff;
	}

	return sqrt(res);
}

void Optimizer::step()
{
	for(const auto &v: m_network_weights)
	{
		v->m_vel = m_momentum * v->m_vel + v->m_diff;
		v->m_value -= m_learning_rate * v->m_vel;
	}
}

void Optimizer::accumulate(const CG::Value &cross_enthropy_loss)
{
	assert(cross_enthropy_loss->m_op == CG::Op::CROSS_ENTHROPY);

	// Topological is deterministic and for two CG::Value with the same graph,
	// it will yield the same order. This is how we can pair every weights from
	// the neural net, with it's loss counterpart
	auto loss_weights = topological_sort(cross_enthropy_loss->m_children);

	assert(loss_weights.size() == m_network_weights.size());

	for(size_t i = 0; i < m_network_weights.size(); ++i)
	{
		m_network_weights[i]->m_diff += loss_weights[i]->m_diff;
	}
}

} // namespace NN