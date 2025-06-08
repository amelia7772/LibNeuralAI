#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include<iostream>
#include<fstream>
#include<cmath>
#include<ctime>
#include<algorithm>
#include<sstream>
#include "data_normalizer.h"
using namespace std;

class Neural_network : Data_normalizer{

public:
   int* NETWORK_LAYER_SIZES;
   int NETWORK_SIZE;

   int number_of_inputs;
   int number_of_outputs;

   double** output;
   double*** weight;
   double** bias;

   enum Activation_function{
	   SIGMOID,
	   RELU,
	   IDENTITY,
	   SOFTPLUS,
	   SOFTMAX
   };

   constexpr const static char* activation_functions[] = {
	   "SIGMOID",
	   "RELU",
	   "IDENTITY",
	   "SOFTPLUS",
	   "SOFTMAX"
   };

   Activation_function* NETWORK_LAYER_ACTIVATION_FUNCTIONS;

   friend class Convolutional_neural_network;

private:
   double** error_signal;
   double** output_derivative;

   double*** previous_weight_change;
   double** previous_bias_change;

public:

   static Neural_network get_neural_network_from_file(string path){
	ifstream file = ifstream(path);
	string biases_and_weights = "";
	if(file.is_open()){
		string line = "";
		int line_number = 0;
		while(getline(file,line)){
			biases_and_weights += line + "\n";
			line_number++;
		}
		file.close();
	}
	else{
	    throw exception();
	}
	return get_neural_network_from_string(biases_and_weights);
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
	   for(int i = 1; i < NETWORK_SIZE;i++)
              for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++)
                 for(int k = 0; k < NETWORK_LAYER_SIZES[i-1];k++){
                         file << "w," << i << ',' << j << ',' << k << ',' << weight[i][j][k];
                         if(i == NETWORK_SIZE - 1 && j == NETWORK_LAYER_SIZES[i] - 1 && k == NETWORK_LAYER_SIZES[i-1] -1)
                                 break;
                         file << endl;
                 }
           file << endl;
           for(int i = 1; i < NETWORK_SIZE;i++)
              for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++){
                      file << "b," << i << ',' << j << ',' << bias[i][j];
                      if(i == NETWORK_SIZE - 1 && j == NETWORK_LAYER_SIZES[i] - 1)
                              break;
                        file << endl;
           }
	   file.close();
	   }
	   else
		throw new exception();
   }

   Neural_network(int* NETWORK_LAYER_SIZES,int NETWORK_SIZE,Activation_function* NETWORK_LAYER_ACTIVATION_FUNCTIONS){
	   this->NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
	   this->NETWORK_SIZE = NETWORK_SIZE;
	   for(int i = 0;i < NETWORK_SIZE;i++){
		   if((NETWORK_LAYER_ACTIVATION_FUNCTIONS[i] == SOFTMAX) && (i != (NETWORK_SIZE - 1))){
				cerr << "error: softmax is set as the activation functione to a Neural network's layer that is notthe output layer" << endl;
				exit(2);
		   }
	   }
	   this->NETWORK_LAYER_ACTIVATION_FUNCTIONS = NETWORK_LAYER_ACTIVATION_FUNCTIONS;
	   number_of_inputs = NETWORK_LAYER_SIZES[0];
	   number_of_outputs = NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];
	   output = new double*[NETWORK_SIZE];

	   for(int i = 0;i < NETWORK_SIZE;i++){
		   output[i] = new double[NETWORK_LAYER_SIZES[i]];
	   }

	   for(int i = 0;i < NETWORK_SIZE;i++){
		for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++)
			output[i][j] = 0;
	   }

	   error_signal = new double*[NETWORK_SIZE];

           for(int i = 0;i < NETWORK_SIZE;i++){
                   error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
           }

           for(int i = 0;i < NETWORK_SIZE;i++){
                for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++)
                        error_signal[i][j] = 0;
           }

	   output_derivative = new double*[NETWORK_SIZE];

           for(int i = 0;i < NETWORK_SIZE;i++){
                   output_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];
           }

           for(int i = 0;i < NETWORK_SIZE;i++){
                for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++)
                        output_derivative[i][j] = 0;
           }

	   weight = new double**[NETWORK_SIZE];

	   for(int i = 0;i < NETWORK_SIZE;i++){
		   weight[i] = new double*[NETWORK_LAYER_SIZES[i]];
		   if(i == 0){
			   for(int j = 0; j < NETWORK_LAYER_SIZES[0];j++)
				   weight[0][j] = new double[0];
		   }
		   else{
			   for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++)
				   weight[i][j] = new double[NETWORK_LAYER_SIZES[i - 1]];
		   }
	   }
	   
	   for(int i = 1;i < NETWORK_SIZE;i++){
		   for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++){
			   for(int k = 0; k < NETWORK_LAYER_SIZES[i - 1];k++){
				   weight[i][j][k] = 0;
			   }
		   }
	   }

	   bias = new double*[NETWORK_SIZE];
	   for(int i = 0;i < NETWORK_SIZE;i++){
		   bias[i] = new double[NETWORK_LAYER_SIZES[i]];
	   }
	   for(int i = 1;i < NETWORK_SIZE;i++){
		   for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++)
			   bias[i][j] = 0;
	   }

	   previous_weight_change = new double**[NETWORK_SIZE];
           for(int i = 0;i < NETWORK_SIZE;i++){
                   previous_weight_change[i] = new double*[NETWORK_LAYER_SIZES[i]];
                   if(i == 0){
                           for(int j = 0; j < NETWORK_LAYER_SIZES[0];j++)
                                   previous_weight_change[0][j] = new double[0];
                   }
                   else{
                           for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++)
                                   previous_weight_change[i][j] = new double[NETWORK_LAYER_SIZES[i - 1]];
                   }
           }
           previous_bias_change = new double*[NETWORK_SIZE];
           for(int i = 0;i < NETWORK_SIZE;i++)
                previous_bias_change[i] = new double[NETWORK_LAYER_SIZES[i]];
   }

   void randomize(double lower_bound, double upper_bound){
           srand(time(NULL));
	   for(int i = 1;i < NETWORK_SIZE;i++){
		   for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++)
		   	for(int k = 0; k < NETWORK_LAYER_SIZES[
i - 1];k++){
		   	weight[i][j][k] = lower_bound + ( ((double)rand() / RAND_MAX) * (upper_bound - lower_bound) );
			}
	   }
   }
   
   double* calculate(double* input){
	   double sum = 0;
	   double soft_max_denominator = 0;
	   for(int i = 0; i < NETWORK_LAYER_SIZES[0]; i++)
		   output[0][i] = input[i];
	   for(int i = 1;i < NETWORK_SIZE;i++){
		   for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++){
			for(int k = 0; k < NETWORK_LAYER_SIZES[i - 1];k++){
				sum += output[i-1][k] * weight[i][j][k];
			}
			sum += bias[i][j];

			if(NETWORK_LAYER_ACTIVATION_FUNCTIONS[i] == SIGMOID){
				output[i][j] = sigmoid(sum);
				output_derivative[i][j] = output[i][j] * (1 - output[i][j]);
			}
			else if(NETWORK_LAYER_ACTIVATION_FUNCTIONS[i] == RELU){
				output[i][j] = ReLU(sum);
                                output_derivative[i][j] = step(sum);
			}
			else if(NETWORK_LAYER_ACTIVATION_FUNCTIONS[i] == IDENTITY){
				output[i][j] = sum;
				output_derivative[i][j] = 1;
			}
			else if(NETWORK_LAYER_ACTIVATION_FUNCTIONS[i] == SOFTPLUS){
				output[i][j] = soft_plus(sum);
                                output_derivative[i][j] = sigmoid(sum);
			}
			else{ //if(i == NETWORK_SIZE - 1)
				output[i][j] = sum;
			}
		   }
	   }
	   if(NETWORK_LAYER_ACTIVATION_FUNCTIONS[NETWORK_SIZE - 1] == SOFTMAX){
		   double max_exponent = output[NETWORK_SIZE - 1][0];
                                for(int j = 1; j < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1]; j++)
                                        max_exponent = max(max_exponent,output[NETWORK_SIZE - 1][j]);
                                for(int j = 0; j < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];j++)
                                        soft_max_denominator += exp(output[NETWORK_SIZE - 1][j] - max_exponent);
                                for(int j = 0; j < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];j++){
                                        output[NETWORK_SIZE - 1][j] = soft_max(output[NETWORK_SIZE - 1][j],soft_max_denominator, max_exponent);
				}
	   }
	   double* result = output[NETWORK_SIZE - 1];
	   return result;
   }

   void set_input_normalization_parameters(double mean,double standard_deviation, double minimum_input, double maximum_input,bool do_round){
	   Data_normalizer::set_input_normalization_parameters(mean,standard_deviation,minimum_input,maximum_input,do_round);
   }

   void set_input_normalization_parameters(double mean,double standard_deviation, double* minimum_input, double* maximum_input, bool do_round){
           Data_normalizer::set_input_normalization_parameters(mean,standard_deviation,number_of_inputs,minimum_input,maximum_input,do_round);
   }

   void train(double* input, double* target, double learning_rate,double momentum = 0,int batch_size = 1)
   {
	   if(do_normalization)
		   normalize_input_data(input);
	   calculate(input);
	   back_propogate_error(target);
	   if(momentum == 0)
	   update_weights(learning_rate,batch_size);
	   else
	   update_weights_with_momentum(learning_rate,momentum,batch_size);
   }

   void train_batch(double** input,double** target,int batch_size,double learning_rate, double momentum){

	for(int x = 0; x < batch_size;x++){
		   
		   if(do_normalization)
			   normalize_input_data(input[x]);

		   calculate(input[x]);
		   if(NETWORK_LAYER_ACTIVATION_FUNCTIONS[NETWORK_SIZE - 1]){
		   for(int i = 0; i < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];i++){
			   error_signal[NETWORK_SIZE - 1][i] = output[NETWORK_SIZE - 1][i] - target[x][i];
                   }
                   for(int i = NETWORK_SIZE - 2; i > 0; i--){
                           for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++){
                                   double sum = 0;
                                   for(int k = 0; k < NETWORK_LAYER_SIZES[i+1]; k++)
                                           sum += weight[i+1][k][j] * error_signal[i+1][k];
                                   error_signal[i][j] = sum * output_derivative[i][j];
                           }
		   }
		   }
		   else{
		   for(int i = 0; i < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];i++)
                   error_signal[NETWORK_SIZE - 1][i] = 2 * (output[NETWORK_SIZE - 1][i] - target[x][i]) * output_derivative[NETWORK_SIZE - 1][i];

		   for(int i = NETWORK_SIZE - 2; i > 0; i--){
                           for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++){
                                   double sum = 0;
                                   for(int k = 0; k < NETWORK_LAYER_SIZES[i+1]; k++)
                                           sum += weight[i+1][k][j] * error_signal[i+1][k];
                                   error_signal[i][j] = sum * output_derivative[i][j];
                           }
                   }
		   }

	   for(int i = 1; i < NETWORK_SIZE;i++){
                   for(int j = 0; j < NETWORK_LAYER_SIZES[i]; j++){
                           for(int k = 0; k < NETWORK_LAYER_SIZES[i-1];k++){
                                double delta_weight = (learning_rate * (output[i-1][k] * error_signal[i][j]) + momentum * previous_weight_change[i][j][k]) / (double) batch_size;
                                weight[i][j][k] -= delta_weight;
                                previous_weight_change[i][j][k] = delta_weight;
                           }
                           double delta_bias = (learning_rate * error_signal[i][j] + momentum * previous_bias_change[i][j]) / (double)batch_size;
                           bias[i][j] -= delta_bias;
                           previous_bias_change[i][j] = delta_bias;
                   }
           }

	}
	clean_previous_weights_and_biases_change();
   }

   double get_loss(double* input,double* target){
	   calculate(input);
	   double loss = 0;
	   if(NETWORK_LAYER_ACTIVATION_FUNCTIONS[NETWORK_SIZE - 1] == SOFTMAX){
		   for(int i = 0; i < number_of_outputs;i++)
			if(target[i] == 1){
		   	loss = - log(output[NETWORK_SIZE - 1][i]);
			break;
			}
	   }
	   else
		for(int i = 0; i < number_of_outputs;i++)
		   loss += (output[NETWORK_SIZE - 1][i] - target[i]) * (output[NETWORK_SIZE - 1][i] - target[i]);
	   return loss;
   }

   string get_weights_and_biases_as_string(){
	   ostringstream strs;

	   strs << "s," << NETWORK_SIZE << endl;
           strs << "l,";
	   for(int i = 0; i < NETWORK_SIZE;i++){
	   strs << NETWORK_LAYER_SIZES[i];
	   if(i < NETWORK_SIZE - 1)
		   strs << ',';
	   }
	   strs << endl;
	   strs << "a,";
           for(int i = 0; i < NETWORK_SIZE;i++){
                strs << Neural_network::activation_functions[NETWORK_LAYER_ACTIVATION_FUNCTIONS[i]];
                if(i < NETWORK_SIZE - 1)
                        strs << ',';
           }
	   strs << endl;
	   for(int i = 1; i < NETWORK_SIZE;i++)
	      for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++)
		 for(int k = 0; k < NETWORK_LAYER_SIZES[i-1];k++){
			 strs << "w," << i << "," << j <<  "," << k << "," << weight[i][j][k];
			 if(i == NETWORK_SIZE - 1 && j == NETWORK_LAYER_SIZES[i] - 1 && k == NETWORK_LAYER_SIZES[i-1] -1)
				 break;
			 strs << endl;
		 }
	   strs << endl;
	   for(int i = 1; i < NETWORK_SIZE;i++)
	      for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++){
		      strs << "b," << i << "," << j << "," << bias[i][j];
		      if(i == NETWORK_SIZE - 1 && j == NETWORK_LAYER_SIZES[i] - 1)
			      break;
		      strs << endl;
	   }

	   return strs.str();
   }

   void delete_brain(){
	   for(int i = 1; i < NETWORK_SIZE;i++){
                   for(int j = 0; j < NETWORK_LAYER_SIZES[i]; j++){
                           delete[] previous_weight_change[i][j];
			   delete[] weight[i][j];
                   }
		   delete[] error_signal[i];
		   delete[] output_derivative[i];
		   delete[] output[i];
		   delete[] weight[i];
		   delete[] bias[i];
                   delete[] previous_weight_change[i];
                   delete[] previous_bias_change[i];
        }
	delete[] error_signal;
	delete[] output_derivative;
	delete[] output;
	delete[] weight;
	delete[] bias;
        delete[] previous_weight_change;
        delete[] previous_bias_change;
	delete[] NETWORK_LAYER_SIZES;
	delete[] NETWORK_LAYER_ACTIVATION_FUNCTIONS;
}

private:

   static double sigmoid(double x){
	return (x >= 0) ? 1.0/(1.0 + exp(-x)) : exp(x)/(1 + exp(x));
   }

   static double ReLU(double x){
	   if(x > 0)
		   return x;
	return 0;
   }

   static double soft_plus(double x){
	if(x >= 24)
		return x;
	return log(1 + exp(x));
   }

   static double step(double x){
	return (x > 0) ? 1 : 0;
   }

   static double soft_max(double x, double* inputs, int amount_of_inputs){
	   double max_exponent = inputs[0];
	   for(int i = 1; i < amount_of_inputs; i++)
		   max_exponent = max(max_exponent,inputs[i]);
	   double denominator = 0;
	   for(int i = 0; i < amount_of_inputs;i++)
		   denominator += exp(inputs[i] - max_exponent);
	   return exp(x - max_exponent) / denominator;
   }

   static double soft_max(double x, double denominator, double max_exponent){
	   return exp(x - max_exponent) / denominator;
   }

public:
   void normalize_input_data(double* input){
	Data_normalizer::normalize_input_data(input,number_of_inputs);
   }

   void clean_previous_weights_and_biases_change(){
	   for(int i = 1; i < NETWORK_SIZE;i++)
		   for(int j = 0; j < NETWORK_LAYER_SIZES[i]; j++){
		   for(int k = 0; k < NETWORK_LAYER_SIZES[i-1];k++)
			   previous_weight_change[i][j][k] = 0;
		   previous_bias_change[i][j] = 0;
		   }
   }

private:
   void back_propogate_error(double* target){
	   if(NETWORK_LAYER_ACTIVATION_FUNCTIONS[NETWORK_SIZE - 1] == Neural_network::SOFTMAX){
                   
		   for(int i = 0; i < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];i++){
			   error_signal[NETWORK_SIZE - 1][i] = output[NETWORK_SIZE - 1][i] - target[i];
		   }
                   for(int i = NETWORK_SIZE - 2;i > 0; i--){
                           for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++){
                                   double sum = 0;
                                   for(int k = 0; k < NETWORK_LAYER_SIZES[i+1]; k++)
                                           sum += weight[i+1][k][j] * error_signal[i+1][k];
                                   error_signal[i][j] = sum * output_derivative[i][j];
                           }
                   }
                   }
                   else{
                   for(int i = 0; i < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];i++)
                   error_signal[NETWORK_SIZE - 1][i] = 2 * (output[NETWORK_SIZE - 1][i] - target[i]) * output_derivative[NETWORK_SIZE - 1][i];

                   for(int i = NETWORK_SIZE - 2; i > 0; i--){
                           for(int j = 0; j < NETWORK_LAYER_SIZES[i];j++){
                                   double sum = 0;
                                   for(int k = 0; k < NETWORK_LAYER_SIZES[i+1]; k++)
                                           sum += weight[i+1][k][j] * error_signal[i+1][k];
                                   error_signal[i][j] = sum * output_derivative[i][j];
                           }
                   }
                   }
   }

   void update_weights(double learning_rate,int batch_size = 1){
	   for(int i = 1; i < NETWORK_SIZE;i++){
		   for(int j = 0; j < NETWORK_LAYER_SIZES[i]; j++){
			   for(int k = 0; k < NETWORK_LAYER_SIZES[i-1];k++){
				   double delta_weight = (learning_rate * output[i-1][k] * error_signal[i][j]) / (double)batch_size;
				   weight[i][j][k] -= delta_weight;
			   }
			   double delta_bias = (learning_rate * error_signal[i][j]) / (double)batch_size;
			   bias[i][j] -= delta_bias;
		   }
	   }
   }

   void update_weights_with_momentum(double learning_rate,double momentum, int batch_size = 1){
	   for(int i = 1; i < NETWORK_SIZE;i++){
                   for(int j = 0; j < NETWORK_LAYER_SIZES[i]; j++){
                           for(int k = 0; k < NETWORK_LAYER_SIZES[i-1];k++){
                                double delta_weight = (learning_rate * (output[i-1][k] * error_signal[i][j]) + momentum * previous_weight_change[i][j][k]) / (double) batch_size;
                                weight[i][j][k] -= delta_weight;
                                previous_weight_change[i][j][k] = delta_weight;
                           }
                           double delta_bias = (learning_rate * error_signal[i][j] + momentum * previous_bias_change[i][j]) / (double)batch_size;
                           bias[i][j] -= delta_bias;
                           previous_bias_change[i][j] = delta_bias;
                   }
           }

   }

   static Neural_network get_neural_network_from_string(string biases_and_weights,int index_of_end_of_second_line,int index_of_end_of_third_line){
	   int network_size = 0;
	string network_size_str = "";
	int i = 2;
	while(biases_and_weights[i] != '\n'){
		if(((int)biases_and_weights[i]) - ((int)'0') >= 0 && ((int)biases_and_weights[i]) - ((int)'0') <= 9)
		network_size_str += biases_and_weights[i];
		i++;
	}
	network_size = stoi(network_size_str);
	int* network_layer_sizes = new int[network_size];
	int layer_count = 0;
	string layer_size = "";
	for(int j = i + 3;j < index_of_end_of_second_line + 1;j++){
		if(biases_and_weights[j] == ',' || biases_and_weights[j] == '\n'){
			network_layer_sizes[layer_count] = stoi(layer_size);
			layer_count++;
			layer_size = "";
		}
		else
			layer_size += biases_and_weights[j];
	}

	Activation_function* network_layer_activation_function = new Activation_function[network_size];

	i = index_of_end_of_third_line + 3;
	layer_count = 0;
	string layer_activation_function = "";
	for(int j = index_of_end_of_second_line + 3;j < index_of_end_of_third_line + 1;j++){
		if(biases_and_weights[j] == ',' || biases_and_weights[j] == '\n'){
			if(layer_activation_function == "SIGMOID")
			network_layer_activation_function[layer_count] = SIGMOID;
			else if(layer_activation_function == "RELU")
			network_layer_activation_function[layer_count] = RELU;
			else if(layer_activation_function == "IDENTITY"){
			network_layer_activation_function[layer_count] = IDENTITY;
			}
			else if(layer_activation_function == "SOFTPLUS"){
			network_layer_activation_function[layer_count] = SOFTPLUS;
			}
			else
			network_layer_activation_function[layer_count] = SOFTMAX;
			layer_count++;
			layer_activation_function = "";
		}
		else
			layer_activation_function += biases_and_weights[j];
	}

	Neural_network brain(network_layer_sizes,network_size,network_layer_activation_function);

	double*** weights = brain.weight;
	double** biases = brain.bias;
	int last_new_line_index = index_of_end_of_third_line;
	int layer_index = 0, neuron_index = 0, prev_neuron_index = 0;
	string layer_index_str = "", neuron_index_str = "", prev_neuron_index_str = "";
	int coma_count = 0;
	string value = "";
	for(int i = index_of_end_of_third_line + 1; i < biases_and_weights.length() - 1;i++){
		if(biases_and_weights[i] == '\n'){
			bool is_weight = (biases_and_weights[last_new_line_index+1] == 'w');
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
					   value += biases_and_weights[j];
				   }
				}
			}
			layer_index = stoi(layer_index_str);
			neuron_index = stoi(neuron_index_str);
			if(is_weight){
			prev_neuron_index = stoi(prev_neuron_index_str);
			weights[layer_index][neuron_index][prev_neuron_index] = stod(value);
			}
			else
				biases[layer_index][neuron_index] = stod(value);
			coma_count = 0;
			layer_index = 0;
		        neuron_index = 0;
			prev_neuron_index = 0;
			layer_index_str = "";
			neuron_index_str = "";
			prev_neuron_index_str = "";
			value = "";
			last_new_line_index = i;
		}
	}

	brain.weight = weights;
	brain.bias = biases;
	return brain;
   }

   static Neural_network get_neural_network_from_string(string biases_and_weights){
	   int index_of_end_of_second_line = 0, index_of_end_of_third_line = 0;
	   int i = 0;
	   while(biases_and_weights[i] != '\n')
		i++;
	   i++;
	   while(biases_and_weights[i] != '\n')
                i++;
	   index_of_end_of_second_line = i;
           i++;
	   while(biases_and_weights[i] != '\n')
                i++;
	   index_of_end_of_third_line = i;
	   return get_neural_network_from_string(biases_and_weights,index_of_end_of_second_line,index_of_end_of_third_line);
   }

};

#endif
