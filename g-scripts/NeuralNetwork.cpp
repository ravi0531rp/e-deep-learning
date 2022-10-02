// NeuralNetwork.hpp
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>

// use typedefs for future ease for changing data types like : float to double
typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

// neural network implementation class!
class NeuralNetwork {
public:
	// constructor
	NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));

	// function for forward propagation of data
	void propagateForward(RowVector& input);

	// function for backward propagation of errors made by neurons
	void propagateBackward(RowVector& output);

	// function to calculate errors made by neurons in each layer
	void calcErrors(RowVector& output);

	// function to update the weights of connections
	void updateWeights();

	// function to train the neural network give an array of data points
	void train(std::vector<RowVector*> data);

	// storage objects for working of neural network
	/*
		use pointers when using std::vector<Class> as std::vector<Class> calls destructor of
		Class as soon as it is pushed back! when we use pointers it can't do that, besides
		it also makes our neural network class less heavy!! It would be nice if you can use
		smart pointers instead of usual ones like this
		*/
	std::vector<RowVector*> neuronLayers; // stores the different layers of out network
	std::vector<RowVector*> cacheLayers; // stores the unactivated (activation fn not yet applied) values of layers
	std::vector<RowVector*> deltas; // stores the error contribution of each neurons
	std::vector<Matrix*> weights; // the connection weights itself
	Scalar learningRate;
};


// constructor of neural network class
NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate)
{
	this->topology = topology;
	this->learningRate = learningRate;
	for (uint i = 0; i < topology.size(); i++) {
		// initialize neuron layers
		if (i == topology.size() - 1)
			neuronLayers.push_back(new RowVector(topology[i]));
		else
			neuronLayers.push_back(new RowVector(topology[i] + 1));

		// initialize cache and delta vectors
		cacheLayers.push_back(new RowVector(neuronLayers.size()));
		deltas.push_back(new RowVector(neuronLayers.size()));

		// vector.back() gives the handle to recently added element
		// coeffRef gives the reference of value at that place
		// (using this as we are using pointers here)
		if (i != topology.size() - 1) {
			neuronLayers.back()->coeffRef(topology[i]) = 1.0;
			cacheLayers.back()->coeffRef(topology[i]) = 1.0;
		}

		// initialize weights matrix
		if (i > 0) {
			if (i != topology.size() - 1) {
				weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
				weights.back()->setRandom();
				weights.back()->col(topology[i]).setZero();
				weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0;
			}
			else {
				weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
				weights.back()->setRandom();
			}
		}
	}
};

void NeuralNetwork::propagateForward(RowVector& input)
{
	// set the input to input layer
	// block returns a part of the given vector or matrix
	// block takes 4 arguments : startRow, startCol, blockRows, blockCols
	neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;

	// propagate the data forward and then
	// apply the activation function to your network
	// unaryExpr applies the given function to all elements of CURRENT_LAYER
	for (uint i = 1; i < topology.size(); i++) {
		// already explained above
		(*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
		neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(std::ptr_fun(activationFunction));
	}
}

void NeuralNetwork::calcErrors(RowVector& output)
{
	// calculate the errors made by neurons of last layer
	(*deltas.back()) = output - (*neuronLayers.back());

	// error calculation of hidden layers is different
	// we will begin by the last hidden layer
	// and we will continue till the first hidden layer
	for (uint i = topology.size() - 2; i > 0; i--) {
		(*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
	}
}


void NeuralNetwork::updateWeights()
{
	// topology.size()-1 = weights.size()
	for (uint i = 0; i < topology.size() - 1; i++) {
		// in this loop we are iterating over the different layers (from first hidden to output layer)
		// if this layer is the output layer, there is no bias neuron there, number of neurons specified = number of cols
		// if this layer not the output layer, there is a bias neuron and number of neurons specified = number of cols -1
		if (i != topology.size() - 2) {
			for (uint c = 0; c < weights[i]->cols() - 1; c++) {
				for (uint r = 0; r < weights[i]->rows(); r++) {
					weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
				}
			}
		}
		else {
			for (uint c = 0; c < weights[i]->cols(); c++) {
				for (uint r = 0; r < weights[i]->rows(); r++) {
					weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
				}
			}
		}
	}
}


void NeuralNetwork::propagateBackward(RowVector& output)
{
	calcErrors(output);
	updateWeights();
}


Scalar activationFunction(Scalar x)
{
	return tanhf(x);
}

Scalar activationFunctionDerivative(Scalar x)
{
	return 1 - tanhf(x) * tanhf(x);
}
// you can use your own code here!


void NeuralNetwork::train(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data)
{
	for (uint i = 0; i < input_data.size(); i++) {
		std::cout << "Input to neural network is : " << *input_data[i] << std::endl;
		propagateForward(*input_data[i]);
		std::cout << "Expected output is : " << *output_data[i] << std::endl;
		std::cout << "Output produced is : " << *neuronLayers.back() << std::endl;
		propagateBackward(*output_data[i]);
		std::cout << "MSE : " << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << std::endl;
	}
}


void ReadCSV(std::string filename, std::vector<RowVector*>& data)
{
	data.clear();
	std::ifstream file(filename);
	std::string line, word;
	// determine number of columns in file
	getline(file, line, '\n');
	std::stringstream ss(line);
	std::vector<Scalar> parsed_vec;
	while (getline(ss, word, ', ')) {
		parsed_vec.push_back(Scalar(std::stof(&word[0])));
	}
	uint cols = parsed_vec.size();
	data.push_back(new RowVector(cols));
	for (uint i = 0; i < cols; i++) {
		data.back()->coeffRef(1, i) = parsed_vec[i];
	}

	// read the file
	if (file.is_open()) {
		while (getline(file, line, '\n')) {
			std::stringstream ss(line);
			data.push_back(new RowVector(1, cols));
			uint i = 0;
			while (getline(ss, word, ', ')) {
				data.back()->coeffRef(i) = Scalar(std::stof(&word[0]));
				i++;
			}
		}
	}
}


void genData(std::string filename)
{
	std::ofstream file1(filename + "-in");
	std::ofstream file2(filename + "-out");
	for (uint r = 0; r < 1000; r++) {
		Scalar x = rand() / Scalar(RAND_MAX);
		Scalar y = rand() / Scalar(RAND_MAX);
		file1 << x << ", " << y << std::endl;
		file2 << 2 * x + 10 + y << std::endl;
	}
	file1.close();
	file2.close();
}


