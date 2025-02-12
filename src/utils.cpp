#include "utils.hpp"
#include <random>

void dfs(
	const CG::Value& node,
	std::unordered_set<CG::Value>& visited,
	std::stack<CG::Value>& order )
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


std::vector<CG::Value> topological_sort(const std::vector<CG::Value> &initial_nodes)
{
	std::unordered_set<CG::Value> visited;
	std::stack<CG::Value> order;

	for (const auto& node : initial_nodes)
	{
		if (visited.find(node) == visited.end())
		{
			dfs(node, visited, order);
		}
	}

	std::vector<CG::Value> sorted_order;
	while (!order.empty())
	{
		sorted_order.push_back(order.top());
		order.pop();
	}

	return sorted_order;
}

std::vector<uint32_t> generate_permutation(uint32_t size)
{
	std::vector<uint32_t> output(size, 0);

	// set the value
	for(uint32_t i = 0; i < size; ++i)
	{
		output[i] = i;
	}

	// shuffle
	std::mt19937 engine;
	
	for(uint32_t i = 0; i < size; ++i)
	{
		std::uniform_int_distribution<uint32_t> distribution(i, size-1);
		uint32_t new_id = distribution(engine);
		uint32_t tmp = output[i];
		output[i] = output[new_id];
		output[new_id] = tmp;
	}

	return output;
}
