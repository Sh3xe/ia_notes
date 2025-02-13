#include "compute_graph.hpp"
#include "utils.hpp"

#include <cmath>
#include <cassert>
#include <iostream>


namespace CG 
{

constexpr double cross_entropy_epsilon = 1e-4;

CG::CG(double value):
	m_value(value)
{
}

void CG::backprop()
{
	auto sort = topological_sort(m_children);
	m_diff = 1.0;
	backward();

	for(auto node: sort)
	{
		node->backward();
	}
}

void CG::forward()
{
	switch(m_op)
	{
	case Op::ADD:
		assert(m_children.size() > 1);
		m_value = 0.0;
		for(auto &child: m_children)
			m_value += child->value();
		break;
	case Op::MUL:
		assert(m_children.size() == 2);
		m_value = m_children[1]->value() * m_children[0]->value();
		break;
	case Op::SUB:
		assert(m_children.size() == 2);
		m_value = m_children[0]->value() - m_children[1]->value();
		break;
	case Op::SOFTMAX:
	{
		float exp_total = 0.0;
		for(const auto &c: m_children)
		{
			exp_total += exp(c->value());
		}

		m_value = exp(m_children[m_input_index]->m_value) / exp_total;
		break;
	}
	case Op::LEAF:
		break;
	case Op::CROSS_ENTHROPY:
		m_value = -log(m_children[m_input_index]->m_value + cross_entropy_epsilon);
		break;
	case Op::RELU:
		m_value = m_children[0]->m_value > 0.0 ? m_children[0]->m_value: 0.0;
		break;
	default:
		assert(false);
		break;
	}
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
				m_children[i]->m_diff += m_diff * m_value * (1.0 - m_value);
			}
			else
			{
				m_children[i]->m_diff += - m_diff * m_value * m_value * (exp(m_children[i]->value()) / exp(m_children[m_input_index]->value()));
			}
		}
		break;
	case Op::LEAF:
		break;
	case Op::RELU:
		assert(m_children.size() == 1);
		m_children[0]->m_diff += (m_value > 0.0 ? m_diff: 0.0);
		break;
	case Op::CROSS_ENTHROPY:
		m_children[m_input_index]->m_diff -= m_diff / (m_children[m_input_index]->m_value + cross_entropy_epsilon);
		break;
	default: 
		assert(false);
		break;
	}
}

Value value(double val)
{
	return std::make_shared<CG>(val);
}

Value operator+(const Value &left, const Value &right)
{
	auto ptr = std::make_shared<CG>(0.0);

	ptr->m_children.push_back(left);
	ptr->m_children.push_back(right);
	ptr->m_op = Op::ADD;
	ptr->m_value = left->value() + right->value();
	
	return ptr;
}

Value list_add(const std::vector<Value> &input)
{
	auto ptr = std::make_shared<CG>(0.0);

	ptr->m_children = input;
	ptr->m_op = Op::ADD;
	ptr->m_value = 0.0;
	for(const auto &v: input)
		ptr->m_value += v->value();
	
	return ptr;
}

Value operator-(const Value &left, const Value &right)
{
	auto ptr = std::make_shared<CG>(0.0);

	ptr->m_children.push_back(left);
	ptr->m_children.push_back(right);
	ptr->m_op = Op::SUB;
	ptr->m_value = left->value() - right->value();
	
	return ptr;
}

Value operator*(const Value &left, const Value &right)
{
	auto ptr = std::make_shared<CG>(0.0);

	ptr->m_children.push_back(left);
	ptr->m_children.push_back(right);
	ptr->m_op = Op::MUL;
	ptr->m_value = left->value() * right->value();
	
	return ptr;
}

Value relu(const Value &cg)
{
	auto ptr = std::make_shared<CG>(0.0);

	ptr->m_children.push_back(cg);
	ptr->m_op = Op::RELU;
	ptr->m_value = cg->value() > 0.0 ? cg->value(): 0.0;
	
	return ptr;
}

Value cross_entropy(
	uint32_t y_real,
	const std::vector<Value> &logits
)
{
	assert(y_real < logits.size());
	Value loss = std::make_shared<CG>(0.0);

	loss->m_op = Op::CROSS_ENTHROPY;
	loss->m_children = logits;
	loss->m_input_index = y_real;
	loss->m_value = -log(logits[y_real]->m_value + cross_entropy_epsilon);

	return loss;
}

std::vector<Value> softmax( const std::vector<Value> &input )
{
	std::vector<Value> ret;
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
		ptr->m_value = exp(input[i]->m_value) / exp_sum;

		ret.push_back(ptr);
	}
	
	return ret;
}

} // namespace CG