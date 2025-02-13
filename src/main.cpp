#include "compute_graph.hpp"
#include "dataset.hpp"
#include "neural_network.hpp"
#include "optimizer.hpp"
#include "utils.hpp"
#include "img_data.hpp"
#include <chrono>
#include <cmath>
#include <iostream>

int find_prediction(const std::vector<CG::Value> &y_pred )
{
	double max_value = y_pred[0]->value();
	int max_index = 0;

	for(size_t i = 1; i < y_pred.size(); ++i)
	{
		if(y_pred[i]->value() > max_value)
		{
			max_value = y_pred[i]->value();
			max_index = static_cast<int>(i);
		}
	}

	return max_index;
}

void train_and_save_nn()
{
	int epochs = 100;
	int batch_size = 64;
	int test_size = 100;
	int test_every = 10;
	int current_test_id = 0;

	NN::NeuralNet neural_net({
		NN::linear(28*28, 10),
		NN::softmax()
	});

	NN::Optimizer optimizer(neural_net, 0.0001 / (double)batch_size, 0.9);

	auto [X_train, y_train] = load_mnist_digits_train();
	auto [X_test, y_test] = load_mnist_digits_test();
	auto permutation = generate_permutation(X_train.size());

	for(int epoch = 0; epoch < epochs; ++epoch)
	{
		optimizer.zero_grad();
		
		for(int i = 0; i < batch_size; ++i)
		{
			// Forward Pass
			size_t index = permutation[(epoch*batch_size+i)%X_train.size()];
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

		if(epoch % test_every != 0)
			continue;

		// Test
		double error = 0.0;
		double correct_guess = 0.0;
		for(int i = 0; i < test_size; ++i)
		{
			auto y_pred = neural_net.forward(X_test[current_test_id]);
			auto y_real = y_test[current_test_id];
			auto prediction = find_prediction(y_pred);

			if(static_cast<int>(y_real) == prediction)
				correct_guess += 1.0;

			double err = CG::cross_entropy(y_real, y_pred)->value();
			error += err;
			current_test_id = (++current_test_id) % X_test.size();
		}

		std::cout << "-------------------" << std::endl;
		std::cout << "Epoch " << epoch << " / " << epochs << std::endl;
		std::cout << "Mean error: " << error / (double)test_size<< std::endl;
		std::cout << "Accuracy: " << (correct_guess / (double)test_size)*100 << "%" <<std::endl;
		std::cout << "Gradient L2 norm: " << optimizer.grad_l2_norm() << std::endl;
	}
}

void test_img()
{
	auto img = load_images("../dataset/t10k-images.idx3-ubyte");
	auto label = load_labels("../dataset/t10k-labels.idx1-ubyte");
	for(int i = 0; i < 10; ++i)
	{
		std::string name = std::to_string(i) + " " + std::to_string((int)label[i]) + std::string(".pgm");
		img[i].save_pgm(name);
	}
}

int main()
{
	train_and_save_nn();
	return 0;
}