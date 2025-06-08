#ifndef CONVOLUTIONAL_NEURAL_NETWORK
#define CONVOLUTIONAL_NEURAL_NETWORK

#include<iostream>
#include<fstream>
#include<cmath>
#include<ctime>
#include<algorithm>
#include<stdexcept>
#include <limits>
#include "neural_network.h"
#include "data_normalizer.h"
using namespace std;

class Convolutional_neural_network : Data_normalizer{
public:
	int* NETWORK_LAYER_SIZES;
	int NETWORK_SIZE;
	Neural_network::Activation_function* NETWORK_LAYER_ACTIVATION_FUNCTIONS;

	enum Pooling_function{
		MIN_POOLING,
		MAX_POOLING,
		AVERAGE_POOLING
	};

	constexpr const static char* pooling_functions[] = {
		"MIN_POOLING",
		"MAX_POOLING",
		"AVERAGE_POOLING"
	};
	Pooling_function* NETWORK_LAYER_POOLING_FUNCTIONS;

	int number_of_inputs;
	int final_number_of_outputs;

	double*** output;
	double**** weight;
	double*** bias;
private:
	int number_of_outputs;
	double*** raw_output;
	double*** output_derivative;
	double*** error_signal;
	double**** previous_weight_change;
	double*** previous_bias_change;

	Neural_network* pcnn; //post_convolution_neural_network

	int* kernel_size;
	int* raw_output_size;
	int* output_size;
	int input_dimention_length;
	int stride;
public:
	static Convolutional_neural_network get_convolutional_neural_network_from_file(string path){
	ifstream file = ifstream(path);
	string biases_and_weights = "";
	int index_of_end_of_second_line = 0, index_of_end_of_third_line = 0, index_of_pcnn = 0, index_of_end_of_fourth_line = 0, index_of_end_of_fifth_line = 0, index_of_end_of_sixth_line = 0, index_of_end_of_seventh_line = 0;
	if(file.is_open()){
		string line = "";
		int line_number = 0;
		while(getline(file,line)){
			if(line_number == 2)
				index_of_end_of_second_line = biases_and_weights.length() - 1;
			else if(line_number == 3)
				index_of_end_of_third_line = biases_and_weights.length() - 1;
			else if(line_number == 4)
				index_of_end_of_fourth_line = biases_and_weights.length() - 1;
			else if(line_number == 5)
				index_of_end_of_fifth_line = biases_and_weights.length() - 1;
			else if(line_number == 6)
				index_of_end_of_sixth_line = biases_and_weights.length() - 1;
			else if(line_number == 7)
				index_of_end_of_seventh_line = biases_and_weights.length() - 1;
			else if(line == "PCNN")
				index_of_pcnn = biases_and_weights.length() + line.length() + 1;
			biases_and_weights += line + "\n";
			line_number++;
		}
		file.close();
	}
	else{
	    throw exception();
	}
	string pcnn_biases_and_weights = "";
	for(int i = index_of_pcnn; i < biases_and_weights.length();i++)
		pcnn_biases_and_weights += biases_and_weights[i];
	string temp = "";
	for(int i = 0; i < index_of_pcnn - 6;i++)
		temp += biases_and_weights[i];

	biases_and_weights = temp;
	
	int network_size = 0;
	string network_size_str = "";
	int i = 2;
	while(biases_and_weights[i] != '\n'){
		network_size_str += biases_and_weights[i];
		i++;
	}
	network_size = stoi(network_size_str);
	int* network_layer_sizes = new int[network_size];
	int layer_count = 0;
	string layer_size = "";
	for(int j = i + 3;j < index_of_end_of_second_line + 1;j++){
		if(biases_and_weights[j] == ',' || biases_and_weights[j] == '\n' || ((j == index_of_end_of_second_line) && ((layer_size != "") || (biases_and_weights[j] != ',' && biases_and_weights[j] != '\n')))){
			if((j == index_of_end_of_second_line) && (biases_and_weights[j] != ',' && biases_and_weights[j] != '\n'))
				layer_size += biases_and_weights[j];
			network_layer_sizes[layer_count] = stoi(layer_size);
			layer_count++;
			layer_size = "";
		}
		else
			layer_size += biases_and_weights[j];
	}

	Neural_network::Activation_function* network_layer_activation_function = new Neural_network::Activation_function[network_size];

	layer_count = 0;
	string layer_activation_function = "";
	for(int j = index_of_end_of_second_line + 3;j < index_of_end_of_third_line + 1;j++){
		if(biases_and_weights[j] == ',' || biases_and_weights[j] == '\n' || ((j == index_of_end_of_third_line) && ((layer_activation_function != "") || (biases_and_weights[j] != ',' && biases_and_weights[j] != '\n')))){
			if((j == index_of_end_of_third_line) && (biases_and_weights[j] != ',' && biases_and_weights[j] != '\n'))
				layer_activation_function += biases_and_weights[j];
			if(layer_activation_function == "SIGMOID")
			network_layer_activation_function[layer_count] = Neural_network::SIGMOID;
			else if(layer_activation_function == "RELU")
			network_layer_activation_function[layer_count] = Neural_network::RELU;
			else if(layer_activation_function == "IDENTITY"){
				network_layer_activation_function[layer_count] = Neural_network::IDENTITY;
			}
			else if(layer_activation_function == "SOFTPLUS"){
			network_layer_activation_function[layer_count] = Neural_network::SOFTPLUS;
			}
			else
			network_layer_activation_function[layer_count] = Neural_network::SOFTMAX;
			layer_count++;
			layer_activation_function = "";
		}
		else
			layer_activation_function += biases_and_weights[j];
	}

	string input_dimention_length_str = "";
	for(int j = index_of_end_of_third_line + 3;j < index_of_end_of_fourth_line + 1;j++)
		input_dimention_length_str += biases_and_weights[j];
	int input_dimention_length = stoi(input_dimention_length_str);

	string stride_str = "";
	for(int j = index_of_end_of_fourth_line + 3;j < index_of_end_of_fifth_line + 1;j++)
		stride_str += biases_and_weights[j];
	int stride = stoi(stride_str);

	int* kernel_size = new int[network_size - 1];
	layer_count = 0;
	string layer_kernel_size = "";
	for(int j = index_of_end_of_fifth_line + 3;j < index_of_end_of_sixth_line + 1;j++){
		if(biases_and_weights[j] == ',' || biases_and_weights[j] == '\n' || ((j == index_of_end_of_sixth_line) && ((layer_kernel_size != "") || (biases_and_weights[j] != ',' && biases_and_weights[j] != '\n') )) ){
			if((j == index_of_end_of_sixth_line) && (biases_and_weights[j] != ',' && biases_and_weights[j] != '\n'))
				layer_kernel_size += biases_and_weights[j];
			kernel_size[layer_count] = stoi(layer_kernel_size);
			layer_count++;
			layer_kernel_size = "";
		}
		else
			layer_kernel_size += biases_and_weights[j];
	}

	layer_count = 0;
	string layer_pooling_function = "";
	Pooling_function* network_layer_pooling_functions = new Pooling_function[network_size];
	for(int j = index_of_end_of_sixth_line + 3;j < index_of_end_of_seventh_line + 1;j++){
		if(biases_and_weights[j] == ',' || biases_and_weights[j] == '\n' || ((j == index_of_end_of_seventh_line) && ((layer_pooling_function != "") || (biases_and_weights[j] != ',' && biases_and_weights[j] != '\n') ))){

			if((j == index_of_end_of_seventh_line) && (biases_and_weights[j] != ',' && biases_and_weights[j] != '\n'))
                          layer_pooling_function += biases_and_weights[j];
			if(layer_pooling_function == "MIN_POOLING")
				network_layer_pooling_functions[layer_count] = MIN_POOLING;
			else if(layer_pooling_function == "MAX_POOLING")
				network_layer_pooling_functions[layer_count] = MAX_POOLING;
			else
				network_layer_pooling_functions[layer_count] = AVERAGE_POOLING;
			layer_count++;
			layer_pooling_function = "";
		}
		else
			layer_pooling_function += biases_and_weights[j];
	}

	Neural_network pcnn = Neural_network::get_neural_network_from_string(pcnn_biases_and_weights);

	Convolutional_neural_network brain(input_dimention_length,network_layer_sizes,network_size,kernel_size,stride,network_layer_activation_function,network_layer_pooling_functions,pcnn.NETWORK_LAYER_SIZES,pcnn.NETWORK_SIZE,pcnn.NETWORK_LAYER_ACTIVATION_FUNCTIONS);

	for(int i = 1;i < pcnn.NETWORK_SIZE;i++)
		for(int j = 0;j < pcnn.NETWORK_LAYER_SIZES[i];j++){
			for(int k = 0; k < pcnn.NETWORK_LAYER_SIZES[i-1];k++)
				brain.pcnn->weight[i][j][k] = pcnn.weight[i][j][k];
			brain.pcnn->bias[i][j] = pcnn.bias[i][j];
		}
	
	int last_new_line_index = index_of_end_of_seventh_line;
	int layer_index = 0, neuron_index = 0, prev_neuron_index = 0 ,bias_matrix_index = 0, weight_matrix_index = 0;
	string layer_index_str = "", neuron_index_str = "", prev_neuron_index_str = "", bias_matrix_index_str = "", weight_matrix_index_str = "";
	int coma_count = 0;
	string value = "";
	for(int i = index_of_end_of_seventh_line + 1; i < biases_and_weights.length() - 1;i++){
		if(biases_and_weights[i] == '\n'){
			bool is_weight = (biases_and_weights[last_new_line_index] == 'w');
			for(int j = last_new_line_index + 1; j < i;j++){
				if(biases_and_weights[j] == ','){
					coma_count++;
					continue;
				}
				if(is_weight){
				if(coma_count == 1){
					layer_index_str += biases_and_weights[j];
				}
				else if(coma_count == 2){
					neuron_index_str += biases_and_weights[j];
				}
				else if(coma_count == 3){
					prev_neuron_index_str += biases_and_weights[j];
				}
				else if(coma_count == 4){
					weight_matrix_index_str += biases_and_weights[j];
				}
				else if(coma_count == 5){
					value += biases_and_weights[j];
				}
				}
				else{
				   if(coma_count == 1){
					   layer_index_str += biases_and_weights[j];
				   }
				   else if(coma_count == 2){
					   neuron_index_str += biases_and_weights[j];
				   }
				   else if(coma_count == 3){
					   bias_matrix_index_str += biases_and_weights[j];
				   }
				   else if(coma_count == 4){
					   value += biases_and_weights[j];
				   }
				}
			}
			layer_index = stoi(layer_index_str);
			neuron_index = stoi(neuron_index_str);
			if(is_weight){
			prev_neuron_index = stoi(prev_neuron_index_str);
			weight_matrix_index = stoi(weight_matrix_index_str);
			brain.weight[layer_index][neuron_index][prev_neuron_index][weight_matrix_index] = stod(value);
			}
			else{
				bias_matrix_index = stoi(bias_matrix_index_str);
				try{
				brain.bias[layer_index][neuron_index][bias_matrix_index] = stod(value);
				}
				catch (const std::out_of_range& oor){
					if(value[value.length() - 5] == 'e' && value[value.length() - 5] == '-')
						brain.bias[layer_index][neuron_index][bias_matrix_index] = 0;
					else
						brain.bias[layer_index][neuron_index][bias_matrix_index] = std::numeric_limits<double>::max();
				}
			}
			coma_count = 0;
			layer_index = 0;
		        neuron_index = 0;
			prev_neuron_index = 0;
			weight_matrix_index = 0;
			bias_matrix_index = 0;
			layer_index_str = "";
			neuron_index_str = "";
			prev_neuron_index_str = "";
			weight_matrix_index_str = "";
			bias_matrix_index_str = "";
			value = "";
			last_new_line_index = i + 1;
		}
	}
	return brain;
   }

	void save_to_file(string path){
	   ofstream file = ofstream(path);
	   if(file.is_open()){
	   file << "s," << NETWORK_SIZE << endl;
	   file << "l,";
	   for(int i = 0; i < NETWORK_SIZE;i++){
	   file << NETWORK_LAYER_SIZES[i];
	   if(i < NETWORK_SIZE - 1)
		   file << ',';
	   }
	   file << endl;
	   file << "a,";
	   for(int i = 0; i < NETWORK_SIZE;i++){
	   	file << Neural_network::activation_functions[NETWORK_LAYER_ACTIVATION_FUNCTIONS[i]];
		if(i < NETWORK_SIZE - 1)
			file << ',';
	   }
	   file << endl;
	   file << "d," << input_dimention_length;
	   file << endl;
	   file << "r," << stride;
	   file << endl;
	   file << "k,";
	   for(int i = 1; i < NETWORK_SIZE;i++){
		   file << kernel_size[i];
		   if(i < NETWORK_SIZE - 1)
			   file << ',';
	   }
	   file << endl;
	   file << "p,";
	   for(int i = 0; i < NETWORK_SIZE;i++){
		   file << pooling_functions[NETWORK_LAYER_POOLING_FUNCTIONS[i]];
		   if(i < NETWORK_SIZE - 1)
			   file << ',';
	   }
	   file << endl;

	   for(int i = 1; i < NETWORK_SIZE;i++)
              for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++)
                 for(int k = 0; k < NETWORK_LAYER_SIZES[i-1];k++)
			 for(int x = 0; x < kernel_size[i] * kernel_size[i];x++){
                         file << "w," << i << ',' << j << ',' << k << ',' << x<< ','<< weight[i][j][k][x];
                         file << endl;
           }

           for(int i = 1; i < NETWORK_SIZE;i++)
              for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++)
		      for(int k = 0; k < raw_output_size[i] * raw_output_size[i];k++){
                      file << "b," << i << ',' << j << ',' << k << ','<< bias[i][j][k];
                        file << endl;
           }

	   file << "PCNN";
	   file << endl;
           file << pcnn->get_weights_and_biases_as_string();
	   file.close();
	   }
	   else
		throw new exception();
   }

	Convolutional_neural_network(int input_dimention_length,int* NETWORK_LAYER_SIZES,int NETWORK_SIZE, int* kernel_size,int stride, Neural_network::Activation_function* NETWORK_LAYER_ACTIVATION_FUNCTIONS,Pooling_function* NETWORK_LAYER_POOLING_FUNCTIONS,int* POST_CONVOLUTION_NETWORK_LAYER_SIZES, int POST_CONVOLUTION_NETWORK_SIZE, Neural_network::Activation_function* POST_CONVOLUTION_NETWORK_LAYER_ACTIVATION_FUNCTIONS){
	   this->NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
	   this->NETWORK_SIZE = NETWORK_SIZE;
	   for(int i = 0;i < NETWORK_SIZE;i++){
                   if((NETWORK_LAYER_ACTIVATION_FUNCTIONS[i] == Neural_network::SOFTMAX) && (i != (NETWORK_SIZE - 1))){
                                cerr << "error: softmax is set as the activation functione to a Convolutional neural network's layer that is notthe output layer" << endl;
                                exit(2);
                   }
           }
	   this->NETWORK_LAYER_ACTIVATION_FUNCTIONS = NETWORK_LAYER_ACTIVATION_FUNCTIONS;
	   this->NETWORK_LAYER_POOLING_FUNCTIONS = NETWORK_LAYER_POOLING_FUNCTIONS;
	   number_of_inputs = NETWORK_LAYER_SIZES[0];
	   number_of_outputs = NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];
	   this->stride = stride;

	   this->kernel_size = new int[NETWORK_SIZE];
	   this->kernel_size[0] = 0;
	   for(int i = 1;i < NETWORK_SIZE;i++){
		   this->kernel_size[i] = kernel_size[i-1];
	   }

	   raw_output_size = new int[NETWORK_SIZE];
	   output_size = new int[NETWORK_SIZE];
	   this->input_dimention_length = input_dimention_length;
	   output_size[0] = input_dimention_length;
	   raw_output_size[0] = input_dimention_length;
	   for(int i = 1;i < NETWORK_SIZE;i++){
		raw_output_size[i] = (int)ceil((((double)(output_size[i-1] - this->kernel_size[i]))/this->stride) + 1);
		output_size[i] = (int)ceil((((double)(raw_output_size[i] - 2))/2) + 1);
	   }

	   POST_CONVOLUTION_NETWORK_LAYER_SIZES[0] = number_of_outputs * output_size[NETWORK_SIZE - 1] * output_size[NETWORK_SIZE - 1];
	   pcnn = new Neural_network(POST_CONVOLUTION_NETWORK_LAYER_SIZES,POST_CONVOLUTION_NETWORK_SIZE,POST_CONVOLUTION_NETWORK_LAYER_ACTIVATION_FUNCTIONS);

	   final_number_of_outputs = pcnn->number_of_outputs;

	   output = new double**[NETWORK_SIZE];

	   for(int i = 0;i < NETWORK_SIZE;i++){
		   output[i] = new double*[NETWORK_LAYER_SIZES[i]];
		   for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++)
			   output[i][j] = new double[output_size[i] * output_size[i]];
	   }

	   this->raw_output = new double**[NETWORK_SIZE];

           for(int i = 0;i < NETWORK_SIZE;i++){
                   this->raw_output[i] = new double*[NETWORK_LAYER_SIZES[i]];
                   for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++)
                           this->raw_output[i][j] = new double[raw_output_size[i] * raw_output_size[i]];
           }

	   error_signal = new double**[NETWORK_SIZE];

           for(int i = 0;i < NETWORK_SIZE;i++){
                   error_signal[i] = new double*[NETWORK_LAYER_SIZES[i]];
		   for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++)
			   error_signal[i][j] = new double[raw_output_size[i] * raw_output_size[i]];
           }

	   output_derivative = new double**[NETWORK_SIZE];

           for(int i = 0;i < NETWORK_SIZE;i++){
                   output_derivative[i] = new double*[NETWORK_LAYER_SIZES[i]];
		   for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++)
			   output_derivative[i][j] = new double[raw_output_size[i] * raw_output_size[i]];
           }

	   weight = new double***[NETWORK_SIZE];

	   for(int i = 1;i < NETWORK_SIZE;i++){
		   weight[i] = new double**[NETWORK_LAYER_SIZES[i]];
			for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++){
				   weight[i][j] = new double*[NETWORK_LAYER_SIZES[i - 1]];
				   for(int k = 0; k < NETWORK_LAYER_SIZES[i - 1];k++){
					   weight[i][j][k] = new double[this->kernel_size[i] * this->kernel_size[i]];
				   }
			}
	   }

	   bias = new double**[NETWORK_SIZE];
	   for(int i = 1;i < NETWORK_SIZE;i++){
		   bias[i] = new double*[NETWORK_LAYER_SIZES[i]];
		   for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++){
			   bias[i][j] = new double[raw_output_size[i] * raw_output_size[i]];
		   }
	   }

	   previous_weight_change = new double***[NETWORK_SIZE];
           for(int i = 0;i < NETWORK_SIZE;i++){
                   previous_weight_change[i] = new double**[NETWORK_LAYER_SIZES[i]];
                   if(i == 0){
                           for(int j = 0; j < NETWORK_LAYER_SIZES[0];j++)
                                   previous_weight_change[0][j] = new double*[0];
                   }
                   else{
                           for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++){
                                   previous_weight_change[i][j] = new double*[NETWORK_LAYER_SIZES[i - 1]];
				   for(int k = 0; k < NETWORK_LAYER_SIZES[i - 1];k++){
					   previous_weight_change[i][j][k] = new double[this->kernel_size[i] * this->kernel_size[i]];
				   }
			   }
                   }
           }
           previous_bias_change = new double**[NETWORK_SIZE];
           for(int i = 0;i < NETWORK_SIZE;i++){
                previous_bias_change[i] = new double*[NETWORK_LAYER_SIZES[i]];
		for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++){
			previous_bias_change[i][j] = new double[raw_output_size[i] * raw_output_size[i]];
		}
	   }

	}

	double* calculate(double** input){
           for(int i = 0; i < NETWORK_LAYER_SIZES[0]; i++)
	   	for(int j = 0; j < input_dimention_length * input_dimention_length;j++)
                   	output[0][i][j] = input[i][j];
	   int output_matrix_index = 0;
           for(int i = 1;i < NETWORK_SIZE;i++){
		   int prev_output_size = output_size[i-1];
		   int current_kernel_size = kernel_size[i];
		   int current_kernel_size_squared = current_kernel_size * current_kernel_size;
		   int column_size = prev_output_size - current_kernel_size - 1;
                   for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++){
			double* sum = new double[raw_output_size[i] * raw_output_size[i]];
			for(int k = 0; k < NETWORK_LAYER_SIZES[i - 1];k++){
			double* prev_output_matrix_pointer = output[i-1][k];
			double* current_weight_matrix_pointer = weight[i][j][k];
		        for(int a = 0 ;a < column_size;a++){
			for(int b = 0; b < column_size;b+=stride){
			for(int x = 0; x < current_kernel_size;x++)
				for(int y = 0; y < current_kernel_size;y++)
					sum[output_matrix_index] += prev_output_matrix_pointer[x*prev_output_size + y + a*prev_output_size + b] * current_weight_matrix_pointer[x*current_kernel_size + y];
			sum[output_matrix_index] /= current_kernel_size_squared;
			output_matrix_index++;
			}
			}

			output_matrix_index = 0;
			
			}
			for(int x = 0; x < raw_output_size[i] * raw_output_size[i];x++){
			sum[x] += bias[i][j][x];

                        if(NETWORK_LAYER_ACTIVATION_FUNCTIONS[i] == Neural_network::SIGMOID){
                                raw_output[i][j][x] = Neural_network::sigmoid(sum[x]);
                                output_derivative[i][j][x] = output[i][j][x] * (1 - output[i][j][x]);
                        }
                        else if(NETWORK_LAYER_ACTIVATION_FUNCTIONS[i] == Neural_network::RELU){
                                raw_output[i][j][x] = Neural_network::ReLU(sum[x]);
                                output_derivative[i][j][x] = Neural_network::step(sum[x]);
                        }
                        else if(NETWORK_LAYER_ACTIVATION_FUNCTIONS[i] == Neural_network::IDENTITY){
                                raw_output[i][j][x] = sum[x];
                                output_derivative[i][j][x] = 1;
                        }
                        else{
                                raw_output[i][j][x] = Neural_network::soft_plus(sum[x]);
                                output_derivative[i][j][x] = Neural_network::sigmoid(sum[x]);
                        }
			sum[x] = 0;
			}
			output_matrix_index = 0;
			for(int k = 0; k < NETWORK_LAYER_SIZES[i];k++){
			for(int a = 0 ;a < raw_output_size[i] - 1;a+=2)
			for(int b = 0; b < raw_output_size[i] - 1;b+=2){
			for(int x = 0; x < 2;x++)
                        for(int y = 0; y < 2;y++){
				if(NETWORK_LAYER_POOLING_FUNCTIONS[i] == MIN_POOLING){
				if((x*raw_output_size[i] + y + a*raw_output_size[i] + b) == 0)
					sum[output_matrix_index] = raw_output[i][k][x*raw_output_size[i] + y + a*raw_output_size[i] + b];
				else
                                sum[output_matrix_index] = min(sum[output_matrix_index],raw_output[i][k][x*raw_output_size[i] + y + a*raw_output_size[i] + b]);
                        	}
                        	else if(NETWORK_LAYER_POOLING_FUNCTIONS[i] == MAX_POOLING){
				if((x*raw_output_size[i] + y + a*raw_output_size[i] + b) == 0)
                                        sum[output_matrix_index] = raw_output[i][k][x*raw_output_size[i] + y + a*raw_output_size[i] + b];
                                else
					sum[output_matrix_index] = max(sum[output_matrix_index],raw_output[i][k][x*raw_output_size[i] + y + a*raw_output_size[i] + b]);
                        	}
                        	else{
                                sum[output_matrix_index] += (raw_output[i][k][x*raw_output_size[i] + y + a*raw_output_size[i] + b]) / 4;
				}
			}
				output[i][j][output_matrix_index] = sum[output_matrix_index];
				output_matrix_index++;
			}
			output_matrix_index = 0;
		   	
			}
			delete[] sum;
			}
           }
           double* pcnn_input = new double[number_of_outputs * output_size[NETWORK_SIZE - 1] * output_size[NETWORK_SIZE - 1]];
	   for(int i = 0; i < number_of_outputs;i++)
		   for(int j = 0;j < output_size[NETWORK_SIZE - 1] * output_size[NETWORK_SIZE - 1];j++)
			   pcnn_input[(i * output_size[NETWORK_SIZE - 1] * output_size[NETWORK_SIZE - 1]) + j] = output[NETWORK_SIZE - 1][i][j];
	   double* result = pcnn->calculate(pcnn_input);
	   delete[] pcnn_input;
           return result;
	}

	void randomize(double lower_bound, double upper_bound){
           srand(time(NULL));
           for(int i = 1;i < NETWORK_SIZE;i++)
                   for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++)
                        for(int k = 0; k < NETWORK_LAYER_SIZES[i - 1];k++)
				for(int x = 0; x < kernel_size[i] * kernel_size[i];x++)
                        		weight[i][j][k][x] = lower_bound + ( ((double)rand() / RAND_MAX) * (upper_bound - lower_bound) );
	   pcnn->randomize(lower_bound,upper_bound);
   	}

	void train(double** input, double* target, double learning_rate, double momentum = 0){
		double*** temp_input = new double**[1];
		double** temp_target = new double*[1];
		temp_target[0] = target;
		temp_input[0] = input;
		train_batch(temp_input,temp_target,1,learning_rate,momentum);
		delete[] temp_input;
		delete[] temp_input;
	}

	void train_batch(double*** input,double** target,int batch_size,double learning_rate, double momentum = 0){
		for(int batch_index = 0; batch_index < batch_size;batch_index++){
		calculate(input[batch_index]);
		double* pcnn_input = new double[number_of_outputs * output_size[NETWORK_SIZE - 1] * output_size[NETWORK_SIZE - 1]];
		for(int i = 0; i < number_of_outputs;i++)
			for(int j = 0;j < output_size[NETWORK_SIZE - 1] * output_size[NETWORK_SIZE - 1];j++){
			pcnn_input[(i * output_size[NETWORK_SIZE - 1] * output_size[NETWORK_SIZE - 1]) + j] = output[NETWORK_SIZE - 1][i][j];
			}

		pcnn->train(pcnn_input,target[batch_index],learning_rate,momentum,batch_size);
		delete[] pcnn_input;
		double sum = 0;
		for(int j = 0; j < pcnn->NETWORK_LAYER_SIZES[0];j++){
                for(int k = 0; k < pcnn->NETWORK_LAYER_SIZES[1]; k++){
                        sum += pcnn->weight[1][k][j] * pcnn->error_signal[1][k];
                }
		pcnn->error_signal[0][j] = sum;
		}

		double** post_pooling_last_error_signal = new double*[number_of_outputs];
		for(int i = 0; i < number_of_outputs;i++){
		post_pooling_last_error_signal[i] = new double[output_size[NETWORK_SIZE - 1] * output_size[NETWORK_SIZE - 1]];
		for(int x = 0; x < output_size[NETWORK_SIZE - 1] * output_size[NETWORK_SIZE - 1];x++)
			post_pooling_last_error_signal[i][x] = pcnn->error_signal[0][i*output_size[NETWORK_SIZE - 1]*output_size[NETWORK_SIZE - 1] + x];

		for(int j = 0; j < raw_output_size[NETWORK_SIZE - 1];j++)
			for(int k = 0; k < raw_output_size[NETWORK_SIZE - 1];k++){
			if(NETWORK_LAYER_POOLING_FUNCTIONS[NETWORK_SIZE - 1] == MAX_POOLING || NETWORK_LAYER_POOLING_FUNCTIONS[NETWORK_SIZE - 1] == MIN_POOLING)
				error_signal[NETWORK_SIZE - 1][i][j*raw_output_size[NETWORK_SIZE - 1] + k] = ((output[NETWORK_SIZE - 1][i][(int)((double)ceil(((double)j)/2)) * raw_output_size[NETWORK_SIZE - 1] + (int)((double)ceil(((double)k)/2))] == raw_output[NETWORK_SIZE - 1][i][j*raw_output_size[NETWORK_SIZE - 1] + k]) ? 1:0)* post_pooling_last_error_signal[i][(int)((double)ceil(((double)j)/2)) * raw_output_size[NETWORK_SIZE - 1] + (int)((double)ceil(((double)k)/2))];
			if(NETWORK_LAYER_POOLING_FUNCTIONS[NETWORK_SIZE - 1] == AVERAGE_POOLING)
			error_signal[NETWORK_SIZE - 1][i][j*raw_output_size[NETWORK_SIZE - 1] + k] = 0.25 * post_pooling_last_error_signal[i][(int)((double)ceil(((double)j)/2)) * raw_output_size[NETWORK_SIZE - 1] + (int)((double)ceil(((double)k)/2))];
		}
		delete[] post_pooling_last_error_signal[i];
		}
		delete[] post_pooling_last_error_signal;

		int output_matrix_index = 0;

		for(int i = NETWORK_SIZE - 2;i > 0;i--){
		double** post_pooling_error_signal = new double*[NETWORK_LAYER_SIZES[i]];
		for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++){
			post_pooling_error_signal[j] = new double[output_size[i] * output_size[i]];
			for(int k = 0; k < NETWORK_LAYER_SIZES[i + 1];k++){
                        for(int a = 0 ;a < raw_output_size[i+1] - (kernel_size[i+1]-1);a++){
                        for(int b = 0; b < raw_output_size[i+1] - (kernel_size[i+1]-1);b+=stride){
                        for(int x = 0; x < kernel_size[i+1];x++)
                                for(int y = 0; y < kernel_size[i+1];y++)
                                        post_pooling_error_signal[j][output_matrix_index] += error_signal[i + 1][k][x*raw_output_size[i+1] + y + a*raw_output_size[i+1] + b] * weight[i+1][k][j][x*kernel_size[i+1] + y] * output_derivative[i + 1][k][x*raw_output_size[i+1] + y + a*raw_output_size[i+1] + b];
                        output_matrix_index++;
                        }

                        }
			
                        output_matrix_index = 0;
			
                        }
			
			for(int k = 0; k < raw_output_size[i];k++)
			for(int x = 0; x < raw_output_size[i];x++){
			if(NETWORK_LAYER_POOLING_FUNCTIONS[NETWORK_SIZE - 1] == MAX_POOLING || NETWORK_LAYER_POOLING_FUNCTIONS[NETWORK_SIZE - 1] == MIN_POOLING)
				error_signal[i][j][k*raw_output_size[i] + x] = ((output[i][j][(int)((double)ceil(((double)k)/2)) * raw_output_size[i] + (int)((double)ceil(((double)x)/2))] == raw_output[i][j][k*raw_output_size[i] + x]) ? 1:0)* post_pooling_last_error_signal[j][(int)((double)ceil(((double)k)/2)) * raw_output_size[i] + (int)((double)ceil(((double)x)/2))];
			else if(NETWORK_LAYER_POOLING_FUNCTIONS[NETWORK_SIZE - 1] == AVERAGE_POOLING)
				error_signal[i][j][k*raw_output_size[i] + x] = 0.25 * post_pooling_last_error_signal[j][(int)((double)ceil(((double)k)/2)) * raw_output_size[i] + (int)((double)ceil(((double)x)/2))];
			}
			delete[] post_pooling_error_signal[j];

		}
		delete[] post_pooling_error_signal;
		}

		output_matrix_index = 0;
		for(int i = 1; i < NETWORK_SIZE;i++)
		for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++){
			double delta_weight = 0, delta_bias = 0;
                for(int k = 0; k < NETWORK_LAYER_SIZES[i-1];k++){
                for(int a = 0; a < output_size[i-1] - (raw_output_size[i]-1);a++){
                for(int b = 0; b < output_size[i-1] - (raw_output_size[i]-1);b+=stride){
		for(int x = 0; x < raw_output_size[i];x++)
			for(int y = 0; y < raw_output_size[i];y++)
				delta_weight += output[i-1][k][x*output_size[i-1] + y + a*output_size[i-1] + b] * error_signal[i][j][x*raw_output_size[i] + y];
			delta_weight /= kernel_size[i] * kernel_size[i];
                        delta_weight *= learning_rate;
                        delta_weight += momentum * previous_weight_change[i][j][k][output_matrix_index];
			delta_weight /= (double)batch_size;
			weight[i][j][k][output_matrix_index] -= delta_weight;
                        previous_weight_change[i][j][k][output_matrix_index] = delta_weight;
			output_matrix_index++;
                        }
                        }
		
                        output_matrix_index = 0;
                }

		for(int x = 0; x < raw_output_size[i];x++)
		for(int y = 0; y < raw_output_size[i];y++)
			delta_bias += error_signal[i][j][x*raw_output_size[i] + y] * output_derivative[i][j][x*raw_output_size[i] + y];
		delta_bias *= learning_rate;
		delta_bias += momentum * previous_bias_change[i][j][0];
		delta_bias /= (double)batch_size;
		for(int x = 0; x < raw_output_size[i] * raw_output_size[i];x++)
			bias[i][j][x] -= delta_bias;
		previous_bias_change[i][j][0] = delta_bias;
		}
	}

}

	void set_input_normalization_parameters(double mean,double standard_deviation, double minimum_input, double maximum_input,bool do_round){
		Data_normalizer::set_input_normalization_parameters(mean,standard_deviation,minimum_input,maximum_input,do_round);
	}

	void set_input_normalization_parameters(double mean,double standard_deviation, double* minimum_input, double* maximum_input, bool do_round){
		Data_normalizer::set_input_normalization_parameters(mean,standard_deviation,number_of_inputs,minimum_input,maximum_input,do_round);
	}
	
	void normalize_input_data(double* input){
		Data_normalizer::normalize_input_data(input,number_of_inputs);
	}

	void normalize_input_data(double** input){
		double* input_one_dimentional = new double[output_size[0] * output_size[0]];
		for(int i = 0;i < output_size[0];i++)
			for(int j = 0;j < output_size[0];i++)
				input_one_dimentional[i*output_size[0] + j] = input[i][j];
		normalize_input_data(input_one_dimentional);
		for(int i = 0;i < output_size[0];i++)
			for(int j = 0;j < output_size[0];i++)
				input[i][j] = input_one_dimentional[i*output_size[0] + j];
		delete[] input_one_dimentional;
	}

	double get_loss(double** input,double* target){
		calculate(input);
		double loss = 0;
		if(pcnn->NETWORK_LAYER_ACTIVATION_FUNCTIONS[pcnn->NETWORK_SIZE - 1] == Neural_network::SOFTMAX){
			for(int i = 0; i < pcnn->number_of_outputs;i++){
			     if(target[i] == 1){
			     loss = -log(pcnn->output[pcnn->NETWORK_SIZE - 1][i]);
			     break;
			     }
			}
		}
		else
		for(int i = 0; i < pcnn->number_of_outputs;i++)
			loss += (pcnn->output[pcnn->NETWORK_SIZE - 1][i] - target[i]) * (pcnn->output[pcnn->NETWORK_SIZE - 1][i] - target[i]);
		return loss;
	}

	void delete_brain(){
		for(int i = 0;i < NETWORK_SIZE;i++){
                   for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++){
                                   for(int k = 0; k < NETWORK_LAYER_SIZES[i - 1];k++){
					   if(i > 0){
                                           delete[] weight[i][j][k];
					   delete[] previous_weight_change[i][j][k];
					   }
				   }
				   if(i > 0){
				   delete[] weight[i][j];
				   delete[] bias[i][j];
				   }
				   delete[] output[i][j];
				   delete[] error_signal[i][j];
				   delete[] output_derivative[i][j];
				   if(i > 0){
				   delete[] previous_weight_change[i][j];
				   delete[] previous_bias_change[i][j];
				   }
                           }
		   delete[] output[i];
		   delete[] error_signal[i];
		   delete[] output_derivative[i];
		   if(i > 0){
		   delete[] bias[i];
		   delete[] weight[i];
		   delete[] previous_weight_change[i];
		   delete[] previous_bias_change[i];
		   }
		}
		delete[] kernel_size;
		delete[] output_size;
		delete[] raw_output_size;
		delete[] output;
		delete[] output_derivative;
		delete[] error_signal;
		delete[] bias;
		delete[] weight;
		delete[] previous_weight_change;
		delete[] previous_bias_change;
		delete[] NETWORK_LAYER_SIZES;
		delete[] NETWORK_LAYER_ACTIVATION_FUNCTIONS;
		delete[] NETWORK_LAYER_POOLING_FUNCTIONS;
		pcnn->delete_brain();
		delete pcnn;
	}

};

#endif
