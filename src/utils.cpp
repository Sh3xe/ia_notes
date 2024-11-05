#include "utils.hpp"
#include <random>

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
