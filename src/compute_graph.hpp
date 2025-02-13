#pragma once

#include <memory>
#include <vector>

namespace CG 
{

enum class Op
{
	ADD, SUB, MUL, RELU, SOFTMAX,	CROSS_ENTHROPY,	LEAF
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

	void backward();

	void forward();

	// List of the inputs
	std::vector<std::shared_ptr<CG>> m_children;
	// Operation performed
	Op m_op {Op::LEAF};
	// If the operation is "Softmax", what index of the output of softmax on m_children does this node corresponds to ?
	// If the operation is "CrossEntrhopy", what is the index of the correct class ?
	uint32_t m_input_index {0};
	// The actual value of the node
	double m_value {0.0};
	// The differential of some loss (the called of .backward()) over m_value
	double m_diff {0.0};
	// Used to store the velocity for optimization
	double m_vel {0.0};
};


using Value = std::shared_ptr<CG>;

Value value(double val);

Value operator+(const Value &left, const Value &right);

Value operator-(const Value &left, const Value &right);

Value operator*(const Value &left, const Value &right);

Value relu(const Value &cg);

Value cross_entropy(
	uint32_t y_real,
	const std::vector<Value> &logits
);

std::vector<Value> softmax( const std::vector<Value> &input );

Value list_add(const std::vector<Value> &input);

} // CG