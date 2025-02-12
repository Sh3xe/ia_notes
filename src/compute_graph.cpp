#include "compute_graph.hpp"

#include <cmath>
#include <cassert>
#include <iostream>

namespace CG 
{

constexpr cross_enthropy_epsilon = 1e-4;

CG::CG(double value):
	m_value(value)
{
}

void CG::backprop()
{
	auto sort = topological_sort();
	m_diff = 1.0;
	backward();

	for(auto node: sort)
	{
		node->backward();
	}
}

void CG::zero_grad()
{
	m_diff = 0;
	for(auto &child: m_children)
		child->zero_grad();
}

void dfs(
	const std::shared_ptr<CG>& node,
	std::unordered_set<std::shared_ptr<CG>>& visited,
	std::stack<std::shared_ptr<CG>>& order )
{
	visited.insert(node);
	for (const auto& child : node->m_children)
	{
		if (visited.find(child) == visited.end())
		{
			dfs(child, visited, order);
		}
	}
	order.push(node);
}

std::vector<std::shared_ptr<CG>> CG::topological_sort()
{
	std::unordered_set<std::shared_ptr<CG>> visited;
	std::stack<std::shared_ptr<CG>> order;

	for (const auto& node : m_children)
	{
		if (visited.find(node) == visited.end())
		{
			dfs(node, visited, order);
		}
	}

	std::vector<std::shared_ptr<CG>> sprted_order;
	while (!order.empty())
	{
		sprted_order.push_back(order.top());
		order.pop();
	}

	return sprted_order;
}

void CG::backward()
{
	switch(m_op)
	{
	case Op::ADD:
		for(auto &child: m_children)
			child->m_diff += m_diff;
		break;
	case Op::MUL:
		m_children[0]->m_diff += m_diff * m_children[1]->value();
		m_children[1]->m_diff += m_diff * m_children[0]->value();
		break;
	case Op::SUB:
		m_children[0]->m_diff += m_diff;
		m_children[1]->m_diff -= m_diff;
		break;
	case Op::SOFTMAX:
		for(uint32_t i = 0; i < m_children.size(); ++i)
		{
			if( i == m_input_index)
			{
				m_children[i]->m_diff += m_value * (1.0- m_value);
			}
			else
			{
				m_children[i]->m_diff += m_value * m_value * (exp(m_children[i]->value()) / exp(m_children[m_input_index]->value()));
			}
		}
		break;
	case Op::LEAF:
		break;
	case Op::CROSS_ENTHROPY:
		m_children[m_input_index]->m_diff -= 1 / (m_value + cross_enthropy_epsilon);
		break;
	default: 
		assert(false);
		break;
	}
}

std::shared_ptr<CG> value(double val)
{
	return std::make_shared<CG>(val);
}

std::shared_ptr<CG> operator+(const std::shared_ptr<CG> &left, const std::shared_ptr<CG> &right)
{
	auto ptr = std::make_shared<CG>(0.0);

	ptr->m_children.push_back(left);
	ptr->m_children.push_back(right);
	ptr->m_op = Op::ADD;
	ptr->m_value = left->value() + right->value();
	
	return ptr;
}

std::shared_ptr<CG> operator-(const std::shared_ptr<CG> &left, const std::shared_ptr<CG> &right)
{
	auto ptr = std::make_shared<CG>(0.0);

	ptr->m_children.push_back(left);
	ptr->m_children.push_back(right);
	ptr->m_op = Op::SUB;
	ptr->m_value = left->value() - right->value();
	
	return ptr;
}

std::shared_ptr<CG> operator*(const std::shared_ptr<CG> &left, const std::shared_ptr<CG> &right)
{
	auto ptr = std::make_shared<CG>(0.0);

	ptr->m_children.push_back(left);
	ptr->m_children.push_back(right);
	ptr->m_op = Op::MUL;
	ptr->m_value = left->value() * right->value();
	
	return ptr;
}

std::shared_ptr<CG> relu(const std::shared_ptr<CG> &cg)
{
	auto ptr = std::make_shared<CG>(0.0);

	ptr->m_children.push_back(cg);
	ptr->m_op = Op::RELU;
	ptr->m_value = cg->value() > 0.0 ? cg->value(): 0.0;
	
	return ptr;
}

std::shared_ptr<CG> cross_enthropy(
	uint32_t y_real,
	const std::vector<std::shared_ptr<CG>> &logits
)
{
	assert(y_real < logits.size());
	std::shared_ptr<CG> loss = std::make_shared<CG>(0.0);

	loss->m_op = Op::CROSS_ENTHROPY;
	loss->m_children = logits;
	loss->m_input_index = y_real;
	loss->m_value = -log(logits[y_real] + cross_enthropy_epsilon);

	return loss;
}

const std::vector<std::shared_ptr<CG>> softmax( const std::vector<std::shared_ptr<CG>> &input )
{
	const std::vector<std::shared_ptr<CG>> ret;
	ret.reserve(input.size());

	double exp_sum = 0.0;
	for(const auto &v: input)
		exp_sum += exp(v->m_value);

	for(size_t i = 0; i < input.size(); ++i)
	{
		auto ptr = std::make_shared<CG>(0.0);

		ptr->m_children = input;
		ptr->m_op = Op::SOFTMAX;
		ptr->m_input_index = i;
		ptr->m_value = input[i]->m_value / exp_sum;
	}
	
	return ret;
}


} // 