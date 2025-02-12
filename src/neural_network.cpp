#include "neural_network.hpp"
#include <random>
#include <cassert>

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
}

std::vector<CG::Value> NeuralNet::forward(const std::vector<double> &input)
{

}

std::vector<CG::Value> NeuralNet::construct_tree()
{
	assert(m_architecture.size() != 0);
	size_t input_size = m_architecture.begin()->output_size;

	// convert the input into CG::Value(s)
	std::vector<CG::Value> current_activation;
	current_activation.reserve(input_size);

	for(size_t i = 0; i < input_size; ++i)
	{
		current_activation.push_back(CG::value(0.0));
	}

	// for each layer, apply its input
	std::default_random_engine rng;
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
	for(const auto &layer: m_architecture)
	{
		assert(layer.input_size == current_activation.size());

		std::vector<CG::Value> layer_output;

		// apply the right operation
		switch(layer.operation)
		{
		case Layer::Func::LINEAR:
			layer_output.reserve(layer.output_size);
			for(size_t i = 0; i < layer.output_size; ++i)
			{
				// output = bias + sum_i x_i*w_i
				CG::Value bias = CG::value(distribution(rng));

				std::vector<CG::Value> to_be_added;
				to_be_added.reserve(layer.input_size+1);

				to_be_added.push_back(bias);
				for(auto &v: current_activation)
				{
					to_be_added.push_back( CG::value(distribution(rng)) * v);
				}

				layer_output.push_back(CG::list_add(to_be_added));
			}
			break;
		case Layer::Func::RELU:
			layer_output.reserve(layer.output_size);
			for(const auto &v: current_activation)
			{
				layer_output.push_back(CG::relu(v));
			}
			break;
		case Layer::Func::SOFTMAX:
			layer_output = CG::softmax(current_activation);
			break;
			default:
				assert(false);
				break;
		}

		// switch the 2 lists
		current_activation = layer_output;
	}

	// at the end, the weights are stored as the autograd's graph
	m_weights = current_activation;
}

} // namespace NN