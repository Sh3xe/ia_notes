#include "compute_graph.hpp"
#include "dataset.hpp"
#include "neural_network.hpp"
#include "optimizer.hpp"

#include <iostream>

void train_and_save_nn()
{
	NN::NeuralNet neural_net({
		NN::linear(28*28, 16),
		NN::relu(),
		NN::linear(16, 10),
		NN::softmax()
	});

	NN::Optimizer optimizer(neural_net, 0.01);

	auto [X_train, y_train] = load_mnist_digits_train();
	auto [X_test, y_test] = load_mnist_digits_test();

	int epochs = 1;
	int batch_size = 1;

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