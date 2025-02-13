#include "compute_graph.hpp"
#include "dataset.hpp"
#include "neural_network.hpp"
#include "optimizer.hpp"
#include <chrono>
#include <cmath>

#include <iostream>

void train_and_save_nn()
{
	NN::NeuralNet neural_net({
		NN::linear(28*28, 16),
		NN::relu(),
		NN::linear(16, 10),
		NN::softmax()
	});

	NN::Optimizer optimizer(neural_net, 1e-4);

	auto [X_train, y_train] = load_mnist_digits_train();
	auto [X_test, y_test] = load_mnist_digits_test();

	int epochs = 10;
	int batch_size = 10;
	int test_size = 100;

	for(int epoch = 0; epoch < epochs; ++epoch)
	{
		optimizer.zero_grad();
		
		for(int i = 0; i < batch_size; ++i)
		{
			// Forward Pass
			size_t index = (epoch*batch_size+i)%X_train.size();
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

		// Test
		double error = 0.0;
		for(size_t i = 0; i < test_size; ++i)
		{
			auto y_pred = neural_net.forward(X_test[i]);
			auto y_real = y_test[i];

			double err = CG::cross_entropy(y_real, y_pred)->value();
			error += err;
		}

		std::cout << "Error: " << error << std::endl;
	}

	// neural_net.save("models/mnist_v0");
}

int main()
{
	train_and_save_nn();
	return 0;
}