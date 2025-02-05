#include "compute_graph.hpp"

#include <cmath>
#include <cassert>
#include <iostream>

namespace CG 
{

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

std::vector<std::shared_ptr<CG>> CG::topological_sort()
{
	std::vector<std::shared_ptr<CG>> res = m_children;
	
	// for(uint32_t i = 0; i < res.size(); i++)
	// {
	// 	for(auto child: res[i]->m_children)
	// 		res.push_back(child);
	// }

	assert(false && "not implemented");

	return res;
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
			if( i == m_softmax_index)
			{
				m_children[i]->m_diff += m_value * (1.0- m_value);
			}
			else
			{
				m_children[i]->m_diff += m_value * m_value * (exp(m_children[i]->value()) / exp(m_children[m_softmax_index]->value()));
			}
		}
		break;
	case Op::LEAF:
		break;
	case Op::CROSS_ENTHROPY:
		assert(false && "Not implemented");
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


} // 