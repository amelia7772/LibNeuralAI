#include<iostream>
#include<fstream>
#include<cmath>
#include<ctime>
#include<algorithm>
#include "neural_network.h"
#include "convolutional_neural_network.h"
using namespace std;

int main(){
ifstream training_set_file = ifstream("/data/data/com.termux/files/home/bin/AI/train-images-idx3-ubyte", ios::in | ios::binary);
ifstream labels_file = ifstream("/data/data/com.termux/files/home/bin/AI/train-labels-idx1-ubyte",ios::in | ios::binary);
char* buffer;
int length = 0;
if(training_set_file.is_open()){
	training_set_file.seekg (0, ios::end);
	length = training_set_file.tellg();
	training_set_file.seekg (0, ios::beg);
	buffer = new char[length];
	training_set_file.read(buffer,length);
        training_set_file.close();
}
else{
        return 1;
}

int length_of_data = length / (28*28) - 16;
double** data = new double*[length_of_data];
for(int i = 16; i < length;i+=(28*28)){
    data[(i-16)/(28*28)] = new double[28*28];
    for(int j = 0; j < (28*28);j++){
	data[(i-16)/(28*28)][j] = (int)buffer[i + j];
    }
}

delete[] buffer;

char* labels;
if(labels_file.is_open()){
        labels_file.seekg (0, ios::end);
	length = labels_file.tellg();
        labels_file.seekg (0, ios::beg);
        labels = new char[length];
	labels_file.read(labels,length);
	labels_file.close();
}
else{
        return 1;
}
/*int brain_layer_sizes_array[] = {28*28,392,196,98,10};
int* brain_layer_sizes = brain_layer_sizes_array;
Neural_network::Activation_function brain_activation_functions[] = {Neural_network::SIGMOID,Neural_network::SOFTPLUS,Neural_network::SOFTPLUS,Neural_network::SOFTPLUS,Neural_network::SOFTMAX};
Neural_network brain(brain_layer_sizes, 5,brain_activation_functions);
brain.randomize(-0.00002,0.00002);
*/
//Neural_network brain = Neural_network::get_neural_network_from_file("/data/data/com.termux/files/home/bin/AI/brain.nn");


int pcnn_brain_layer_sizes_array[] = {1,100,10};
int brain_layer_sizes_array[] = {1,32,64,128};
int network_size = 4;
int pcnn_network_size = 3;
int kernel_size_array[] = {5,3,3};
int stride = 1;
int input_dimention_length = 28;
int* pcnn_brain_layer_sizes = pcnn_brain_layer_sizes_array;
int* brain_layer_sizes = brain_layer_sizes_array;
int* kernel_size = kernel_size_array;
Neural_network::Activation_function pcnn_brain_activation_functions_array[] = {Neural_network::IDENTITY,Neural_network::SOFTPLUS,Neural_network::SOFTPLUS};
Neural_network::Activation_function* pcnn_brain_activation_functions = pcnn_brain_activation_functions_array;

Neural_network::Activation_function brain_activation_functions_array[] = {Neural_network::IDENTITY,Neural_network::IDENTITY,Neural_network::IDENTITY,Neural_network::IDENTITY};
Neural_network::Activation_function* brain_activation_functions = brain_activation_functions_array;

Convolutional_neural_network::Pooling_function brain_pooling_functions_array[] = {Convolutional_neural_network::MAX_POOLING,Convolutional_neural_network::MAX_POOLING,Convolutional_neural_network::MAX_POOLING,Convolutional_neural_network::MAX_POOLING};

Convolutional_neural_network::Pooling_function* brain_pooling_functions = brain_pooling_functions_array;

Convolutional_neural_network brain(input_dimention_length,brain_layer_sizes,network_size,kernel_size,stride,brain_activation_functions,brain_pooling_functions,pcnn_brain_layer_sizes,pcnn_network_size,pcnn_brain_activation_functions);

brain.randomize(-0.00003,0.00003);

//Convolutional_neural_network brain = Convolutional_neural_network::get_convolutional_neural_network_from_file("/data/data/com.termux/files/home/bin/AI/brain.cnn");
brain.set_input_normalization_parameters(0.5,0.5,0,255,false);
const int batch_size = 100;
const double learning_rate = 0.001;
const double momentum = 0.9;
double amount_of_error = 0;
random_shuffle(&data[0],&data[length_of_data - 1]);
int first_indices_of_ten_digits[10];
for(int i = 0; i < 10; i++)
	first_indices_of_ten_digits[i] = -1;
for(int i = 0; i < length_of_data; i++){
	if(first_indices_of_ten_digits[labels[i]] < 0)
		first_indices_of_ten_digits[labels[i]] = i;
	bool is_finished = true;
	for(int j = 0; j < 10; j++){
		is_finished = is_finished && (first_indices_of_ten_digits[j] >= 0);
	if(is_finished)
		break;
	}
}

do{
for(int i = 0; i < length_of_data/batch_size;i++){
	double** batch_target_outputs = new double*[batch_size];
	double*** batch_inputs = new double**[batch_size];
	for(int j = 0; j < batch_size;j++){
		batch_inputs[j] = new double*[28];
		for(int k = 0; k < 28;k++)
			batch_inputs[j][k] = new double[28];
		double label = (double)((int)labels[(i * batch_size)+j]);
		//double target_output[] = {(double)((int)label)};
		double target_output[] = { (label == 0.0) ? 1.0:0.0, (label == 1.0) ? 1.0:0.0,(label == 2.0) ? 1.0:0.0,(label == 3.0) ? 1.0:0.0,(label == 4.0) ? 1.0:0.0,(label == 5.0) ? 1.0:0.0,(label == 6.0) ? 1.0:0.0,(label == 7.0) ? 1.0:0.0,(label == 8.0) ? 1.0:0.0,(label == 9.0) ? 1.0:0.0};
		batch_target_outputs[j] = new double[10];
		for(int k = 0; k < 10;k++)
			batch_target_outputs[j][k] = target_output[k];
		for(int k = 0; k < 28;k++)
		for(int x = 0; x < 28;x++)
			batch_inputs[j][k][x] = data[(i * batch_size)+j][k*28 + x];
		
		//brain.train(data[(i*batch_size)+j], target_output,learning_rate,momentum,batch_size);
		/*
                double* output = brain.calculate(data[2]);
                cout << "{ ";
                for(int k = 0; k < brain.number_of_outputs; k++)
		cout << output[k] << " ,";
		cout << "}" << endl;
		*/
      }

	
      brain.train_batch(batch_inputs,batch_target_outputs,batch_size,learning_rate,momentum);

      for(int j = 0; j < batch_size;j++){
	      for(int k = 0; k < 28;k++)
	      delete[] batch_inputs[j][k];
	      delete[] batch_inputs[j];
	      delete[] batch_target_outputs[j];
      }
      delete[] batch_inputs;
      delete[] batch_target_outputs;

      amount_of_error = 0;

      for(int j = 0; j < 10; j++){
      double** temp_input = new double*[28];
      for(int k = 0; k < 28;k++){
	      temp_input[k] = new double[28];
	      for(int x = 0; x < 28;x++)
		      temp_input[k][x] = data[first_indices_of_ten_digits[j]][k*28 + x];
      }

      double label = (double)((int)labels[first_indices_of_ten_digits[j]]);
      double target_output[] = {(label == 0.0) ? 1.0:0.0, (label == 1.0) ? 1.0:0.0,(label == 2.0) ? 1.0:0.0,(label == 3.0) ? 1.0:0.0,(label == 4.0) ? 1.0:0.0,(label == 5.0) ? 1.0:0.0,(label == 6.0) ? 1.0:0.0,(label == 7.0) ? 1.0:0.0,(label == 8.0) ? 1.0:0.0,(label == 9.0) ? 1.0:0.0};
      amount_of_error += brain.get_loss(temp_input/*data[first_indices_of_ten_digits[j]]*/,target_output)/10;
      
      
      for(int k = 0; k < 28;k++)
	      delete[] temp_input[k];
      delete[] temp_input;
      

      if(amount_of_error <= 0.005)
	      break;
      }
      
      double** temp_input = new double*[28];
      for(int k = 0; k < 28;k++){
              temp_input[k] = new double[28];
              for(int x = 0; x < 28;x++)
                      temp_input[k][x] = data[2][k*28 + x];
      }
      

	double* output = brain.calculate(/*data[2]*/temp_input);
	cout << "{ ";
	for(int j = 0; j < brain.final_number_of_outputs; j++){
		cout << output[j];
		if(j < brain.final_number_of_outputs - 1)
			cout << ',';
	}
	cout << '}' << endl;

      
      for(int k = 0; k < 28;k++)
                delete[] temp_input[k];
        delete[] temp_input;
	
      cout << "Loss: " << amount_of_error << endl;
      if((i % 60)==0){
      brain.save_to_file("/data/data/com.termux/files/home/bin/AI/brain.cnn");
      }
      /*
      double* output = brain.calculate(data[2]/\*temp_input*\/);
      cout << "{ ";
      for(int j = 0; j < brain.number_of_outputs; j++){
	      cout << output[j];
	      if(j < brain.number_of_outputs - 1)
		      cout << ',';
      }
      cout << '}' << endl;
      */
      //usleep(1000000 * 4);
}
brain.save_to_file("/data/data/com.termux/files/home/bin/AI/brain.cnn");


      double** temp_input = new double*[28];
      for(int k = 0; k < 28;k++){
              temp_input[k] = new double[28];
              for(int x = 0; x < 28;x++)
                      temp_input[k][x] = data[2][k*28 + x];
      }

      double* output = brain.calculate(/*data[2]*/temp_input);
      cout << "{ ";
      for(int j = 0; j < brain.final_number_of_outputs; j++){
              cout << output[j];
              if(j < brain.final_number_of_outputs - 1)
                      cout << ',';
      }
      cout << '}' << endl;



      
      for(int k = 0; k < 28;k++)
                delete[] temp_input[k];
        delete[] temp_input;

      cout << "Loss: " << amount_of_error << endl;
}while(amount_of_error > 0.005);

cout << "Loss: " << amount_of_error << endl;

for(int i = 0; i < length_of_data;i++)
    delete[] data[i];
delete[] data;
delete[] labels;

ifstream test_data_file = ifstream("/data/data/com.termux/files/home/bin/AI/t10k-images-idx3-ubyte", ios::in | ios::binary);
ifstream test_labels_file = ifstream("/data/data/com.termux/files/home/bin/AI/t10k-labels-idx1-ubyte",ios::in | ios::binary);

int test_length = 17;
char* test_buffer;
if(test_data_file.is_open()){
        test_data_file.seekg (0, ios::end);
        test_length = test_data_file.tellg();
        test_data_file.seekg (0, ios::beg);
        test_buffer = new char[test_length];
        test_data_file.read(test_buffer,length);
        test_data_file.close();
}
else{
        throw exception();
}

int length_of_test_data = test_length / (28*28) - 16;
double** test_data = new double*[length_of_test_data];
for(int i = 16; i < test_length;i+=(28*28)){
    test_data[(i-16)/(28*28)] = new double[28*28];
    for(int j = 0; j < (28*28);j++){
        test_data[(i-16)/(28*28)][j] = (int)test_buffer[i + j];
    }
}
delete[] test_buffer;

char* test_labels;
if(test_labels_file.is_open()){
	test_labels_file.seekg (0, ios::end);
        length = test_labels_file.tellg();
	test_labels_file.seekg(0, ios::beg);
	test_labels = new char[length];
	test_labels_file.read(test_labels,length);
        test_labels_file.close();
}
else{
	throw exception();
}


double average = 0;
for(int i = 0; i < length_of_test_data;i++){
	double label = (double)((int)test_labels[i]);
	//double target_output[] = {(double)((int)label)};
	double target_output[] = {(label == 0.0) ? 1.0:0.0, (label == 1.0) ? 1.0:0.0,(label == 2.0) ? 1.0:0.0,(label == 3.0) ? 1.0:0.0,(label == 4.0) ? 1.0:0.0,(label == 5.0) ? 1.0:0.0,(label == 6.0) ? 1.0:0.0,(label == 7.0) ? 1.0:0.0,(label == 8.0) ? 1.0:0.0,(label == 9.0) ? 1.0:0.0};
	double outputs_average = 0;
	brain.normalize_input_data(test_data[i]);
	
	double** temp_input = new double*[28];
	for(int k = 0; k < 28;k++){
		temp_input[k] = new double[28];
		for(int x = 0; x < 28;x++)
		temp_input[k][x] = test_data[i][k*28 + x];
	}
	
	double* outputs = brain.calculate(/*test_data[i]*/temp_input);
	int max_probability_digit = 0;
	for(int j = 1; j < brain.final_number_of_outputs;j++)
		max_probability_digit = (outputs[j] > outputs[max_probability_digit]) ? j:max_probability_digit;
	bool is_correct = (max_probability_digit == (int)label);
        average += (double)is_correct / length_of_test_data;
	for(int k = 0; k < 28;k++)
		delete[] temp_input[k];
	delete[] temp_input;
}

cout << average << endl;

for(int i = 0; i < length_of_test_data;i++)
    delete[] test_data[i];
delete[] test_data;
delete[] test_labels;
brain.delete_brain();

return 0;
}
