#pragma once
#include <memory>
#include <vector>

namespace CG 
{

enum class Op
{
	ADD,
	SUB,
	MUL,
	RELU,
	SOFTMAX,
	CROSS_ENTHROPY,
	LEAF
};

class CG
{
public:
	CG(double value);

	// Returns the node's value
	inline double value() const {return m_value; }

	// Returns the node's differential over the node which called .backward()
	inline double diff() const {return m_diff; }

	// Initiate backpropagation from this node's value
	void backprop();

	// Zero out all of the gradients starting from the current node
	void zero_grad();

	friend std::shared_ptr<CG> value(double val);
	friend std::shared_ptr<CG> operator+(const std::shared_ptr<CG> &left, const std::shared_ptr<CG> &right);
	friend std::shared_ptr<CG> operator-(const std::shared_ptr<CG> &left, const std::shared_ptr<CG> &right);
	friend std::shared_ptr<CG> operator*(const std::shared_ptr<CG> &left, const std::shared_ptr<CG> &right);
	friend std::shared_ptr<CG> relu(const std::shared_ptr<CG> &cg);

private:
	void backward();

	std::vector<std::shared_ptr<CG>> topological_sort();

	// List of the inputs
	std::vector<std::shared_ptr<CG>> m_children;
	// Operation performed
	Op m_op {Op::LEAF};
	// If the operation is "Softmax", what index of the output of softmax on m_children does this node corresponds to?
	uint32_t m_softmax_index {0};
	// The actual value of the node
	double m_value {0.0};
	// The differential of some loss (the called of .backward()) over m_value
	double m_diff {0.0};
};

std::shared_ptr<CG> value(double val);
std::shared_ptr<CG> operator+(const std::shared_ptr<CG> &left, const std::shared_ptr<CG> &right);
std::shared_ptr<CG> operator-(const std::shared_ptr<CG> &left, const std::shared_ptr<CG> &right);
std::shared_ptr<CG> operator*(const std::shared_ptr<CG> &left, const std::shared_ptr<CG> &right);
std::shared_ptr<CG> relu(const std::shared_ptr<CG> &cg);

} // CG