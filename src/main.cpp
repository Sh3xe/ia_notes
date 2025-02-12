#include "compute_graph.hpp"
#include "img_data.hpp"
#include "neural_network.hpp"
#include "optimizer.hpp"

#include <iostream>

std::vector<std::vector<double>> fetch_mnist_train_images()
{
	auto images = load_images( "../dataset/train-images.idx3-ubyte" );
	std::vector<std::vector<double>> converted_images;

	for(const auto &img: images)
	{
		converted_images.push_back(img.convert_to_01_vector());
	}

	return converted_images;
}

std::vector<uint32_t> fetch_mnist_train_labels()
{
	auto labels = load_labels( "../dataset/train-labels.idx3-ubyte" );
	std::vector<uint32_t> converted_labels;

	for(const auto &label: labels)
	{
		converted_labels.push_back(static_cast<uint32_t>(label));
	}

	return converted_labels;
}

void train_and_save_nn()
{
	NN::NeuralNet neural_net({
		NN::linear(28*28, 16),
		NN::relu(),
		NN::linear(16, 10),
		NN::softmax()
	});

	NN::Optimizer optimizer(neural_net, 0.01);

	auto X_train = fetch_mnist_train_images();
	auto y_train = fetch_mnist_train_labels();

	int epochs = 10;
	int batch_size = 32;

	for(int epoch = 0; epoch < epochs; ++epoch)
	{
		for(int batch = 0; batch < X_train.size() / batch_size; ++batch)
		{
			optimizer.zero_grad();

			for(int i = 0; i < batch_size; ++i)
			{
				// Forward Pass
				size_t index = (batch*batch_size+i)%X_train.size();
				std::vector<CG::Value> logits = neural_net.forward(X_train[index]);

				// Loss function
				CG::Value loss = CG::cross_entropy(y_train[index], logits);

				// Gradient calculation
				loss->backprop();

				// Accumulate the loss of multiple values
				optimizer.accumulate(loss);
			}

			// Gradient descent step
			optimizer.step();
		}
	}

	// neural_net.save("models/mnist_v0");
}

int main()
{
	train_and_save_nn();

	return 0;
}