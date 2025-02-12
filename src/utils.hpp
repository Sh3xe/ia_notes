#pragma once

#include "compute_graph.hpp"

#include <vector>
#include <cstdint>

void dfs(
	const CG::Value& node,
	std::unordered_set<CG::Value>& visited,
	std::stack<CG::Value>& order );

std::vector<CG::Value> topological_sort(const std::vector<CG::Value> &initial_nodes);

std::vector<uint32_t> generate_permutation(uint32_t size);