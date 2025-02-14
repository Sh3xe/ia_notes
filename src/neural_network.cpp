#include "neural_network.hpp"
#include "utils.hpp"
#include <random>
#include <fstream>
#include <iostream>
#include <cassert>
#include <ctime>

static std::string layer_name(NN::Layer::Func func)
{
	switch(func)
	{
	case NN::Layer::Func::LINEAR:
		return "lin";
	case NN::Layer::Func::SOFTMAX:
		return "sm";
	case NN::Layer::Func::RELU:
		return "relu";
	default:
		assert(false);
	}

	return "";
}

NN::Layer::Func layer_func(const std::string &str)
{
	if(str == "lin") 
		return NN::Layer::Func::LINEAR;
	if(str == "sm")
		return NN::Layer::Func::SOFTMAX;
	if(str == "relu")
		return NN::Layer::Func::RELU;

	assert(false);
	return NN::Layer::Func::LINEAR;
}

namespace NN 
{

Layer linear(int input_size, int output_size)
{
	Layer ret;

	ret.input_size = input_size;
	ret.output_size = output_size;
	ret.operation = Layer::Func::LINEAR;

	return ret;
}

Layer relu()
{
	Layer ret;

	ret.input_size = 0;
	ret.output_size = 0;
	ret.operation = Layer::Func::RELU;

	return ret;
}

Layer softmax()
{
	Layer ret;

	ret.input_size = 0;
	ret.output_size = 0;
	ret.operation = Layer::Func::SOFTMAX;

	return ret;
}

NeuralNet::NeuralNet( const std::initializer_list<Layer> &layer_desc )
{
	m_architecture = layer_desc;
	auto [input, output] = construct_tree();
	m_input_weights = input;
	m_output_weights = output;
}

std::vector<CG::Value> NeuralNet::forward(const std::vector<double> &input)
{
	assert(input.size() == m_input_weights.size());

	// Set the input
	for(size_t i = 0; i < input.size(); ++i)
	{
		m_input_weights[i]->m_value = input[i];
	}

	// Propagate the values forward
	auto topo_sort = topological_sort(m_output_weights);
	for(auto it = topo_sort.rbegin(); it != topo_sort.rend(); ++it)
	{
		it->get()->forward();
	}

	// We don't want to copy the values of the NeuralNet and let the user of the function
	// change its weights, so we create a copy of the tree
	auto [_, output] = construct_tree(false);
	auto copy_topo_sort = topological_sort(output);

	assert(copy_topo_sort.size() == topo_sort.size());
	for(size_t i = 0; i < topo_sort.size(); ++i)
	{
		copy_topo_sort[i]->m_value = topo_sort[i]->m_value;
	}

	return output;
}

std::pair<std::vector<CG::Value>, std::vector<CG::Value>> NeuralNet::construct_tree(bool random)
{
	assert(m_architecture.size() != 0);
	size_t input_size = m_architecture.begin()->input_size;

	// convert the input into CG::Value(s)
	std::vector<CG::Value> current_activation;
	current_activation.reserve(input_size);

	for(size_t i = 0; i < input_size; ++i)
	{
		current_activation.push_back(CG::value(0.0));
	}

	auto input_weights = current_activation;

	// for each layer, apply its input
	std::default_random_engine rng;
	rng.seed(time(NULL));
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
	for(const auto &layer: m_architecture)
	{
		std::vector<CG::Value> layer_output;

		// apply the right operation
		switch(layer.operation)
		{
		case Layer::Func::LINEAR:
		{
			assert((size_t)layer.input_size == current_activation.size());
			layer_output.reserve(layer.output_size);
			for(int i = 0; i < layer.output_size; ++i)
			{
				// output = bias + sum_i x_i*w_i
				CG::Value bias = CG::value(random ? distribution(rng): 0.0);

				std::vector<CG::Value> to_be_added;
				to_be_added.reserve(layer.input_size+1);

				to_be_added.push_back(bias);
				for(auto &v: current_activation)
				{
					to_be_added.push_back( CG::value(random ? distribution(rng): 0.0) * v);
				}

				layer_output.push_back(CG::list_add(to_be_added));
			}

			break;
		}
		case Layer::Func::RELU:
		{
			layer_output.reserve(current_activation.size());
			for(const auto &v: current_activation)
			{
				layer_output.push_back(CG::relu(v));
			}
			break;
		}
		case Layer::Func::SOFTMAX:
		{
			layer_output = CG::softmax(current_activation);
			break;
			default:
				assert(false);
				break;
		}
		}

		// switch the 2 lists
		current_activation = layer_output;
	}

	// return [input_weights, output_weights]
	return std::make_pair(input_weights, current_activation);
}


bool NeuralNet::save_weights(const std::string &path)
{
	std::ofstream file {path};

	if(!file)
	{
		std::cout << "Cannot open: \"" << path << "\"" << std::endl;
		return false;
	}

	file << m_architecture.size() << std::endl;

	for(auto layer: m_architecture)
	{
		file << layer_name(layer.operation) << " " << layer.input_size << " " << layer.output_size << std::endl;
	}

	for(const auto &v: topological_sort(m_output_weights))
	{
		file << v->value() << " ";
	}

	file.close();
	return true;
}
	
bool NeuralNet::load_weights(const std::string &path)
{
	std::ifstream file {path};

	if(!file)
	{
		std::cout << "Cannot open: \"" << path << "\"" << std::endl;
		return false;
	}

	int layer_count = 0;
	file >> layer_count;

	m_architecture.clear();
	std::string layer_name;
	int input_size = 0;
	int output_size = 0;
	NN::Layer layer;
	for(int i = 0; i < layer_count; ++i)
	{
		file >> layer_name >> input_size >> output_size;
		layer.operation = layer_func(layer_name);
		layer.input_size = input_size;
		layer.output_size = output_size;

		m_architecture.push_back(layer);
	}

	auto [m_input_weights, m_output_weights] = construct_tree(false);

	for(const auto &v: topological_sort(m_output_weights))
	{
		file >> v->m_value;
	}

	file.close();
	return true;
}

} // namespace NN