#include<iostream>
#include<fstream>
#include<cmath>
#include<ctime>
#include<algorithm>
#include "neural_network.h"
using namespace std;

double calculate_accuracy(Neural_network brain){
	double x,y,z;
    	double* input = new double[3];
	double* target_output = new double[3];
	double sum = 0;
	double average = 0;
   	srand(time(NULL));
	for(int i = 0;i < 100000;i++){
		x = (((rand() % 100) >= 50 ? -1 : 1) * (double)rand() / RAND_MAX) * 100;
		y = (((rand() % 100) >= 50 ? -1 : 1) * (double)rand() / RAND_MAX) * 100;
  		z = (((rand() % 100) >= 50 ? -1 : 1) * (double)rand() / RAND_MAX) * 100;
    		input[0] = x;
    		input[1] = y;
    		input[2] = z;
   		sum = x + y + z;
    		target_output[0] = sum  < 0 ?1.0 : 0.0;
    		target_output[1] = sum > 0 ? 1.0 : 0.0;
    		target_output[2] = sum == 0 ? 1.0 : 0.0;
		double outputs_average = 0;
		double* outputs = brain.calculate(input);
		double is_correct = 1.0;
		for(int i = 0; i < brain.number_of_outputs;i++)
			if(abs(target_output[i] - outputs[i]) >= 0.2){
				is_correct = 0.0;
				break;
			}
		average += is_correct / 100000.0;
		outputs_average = 0;
	}
	delete[] input;
	delete[] target_output;

	return average;
}

int main(){
    int brain_layer_sizes_array[] = {3,100,3};
    int* brain_layer_sizes = brain_layer_sizes_array;
    Neural_network brain(brain_layer_sizes, 3);
    brain.randomize(-0.0001,0.0001);


    Neural_network brain = Neural_network::get_neural_network_from_file("/data/data/com.termux/files/home/bin/brain.nn");
    double x,y,z;
    int batch_size = 10000;
    double** input = new double*[batch_size];
    double** target_output = new double*[batch_size];
    double sum = 0;
    double learning_rate = 0.1;
    double friction = 0.9;
    srand(time(NULL));
    for(int j = 0; j < 400;j++){
    for(int i = 0; i < 10000000/batch_size; i++){
    for(int k = 0; k < batch_size;k++){
    input[k] = new double[3];
    target_output[k] = new double[3];
    x = (((rand() % 100) >= 50 ? -1 : 1) * (double)rand() / RAND_MAX) * 100;
    y = (((rand() % 100) >= 50 ? -1 : 1) * (double)rand() / RAND_MAX) * 100;
    z = (((rand() % 100) >= 50 ? -1 : 1) * (double)rand() / RAND_MAX) * 100;
    input[k][0] = x;
    input[k][1] = y;
    input[k][2] = z;
    sum = x + y + z;
    target_output[k][0] = sum < 0 ? 1.0 : 0.0;
    target_output[k][1] = sum > 0 ? 1.0 : 0.0;
    target_output[k][2] = sum == 0 ? 1.0 : 0.0;
    }
    brain.train_batch(input ,target_output,batch_size,learning_rate,friction);
    double input_display[] = {1,-2,0};
      double* output = brain.calculate(input[batch_size - 1]);
      cout << "{ ";
      for(int i = 0; i < brain.number_of_outputs; i++)
      cout << "} : " << input[batch_size - 1][0] + input[batch_size - 1][1] + input[batch_size - 1][2] << endl;
      for(int k = 0; k < batch_size;k++){
      delete[] input[k];
      delete[] target_output[k];
}
    }
    
    //brain.print_weights_and_biases();

    brain.save_to_file("/data/data/com.termux/files/home/bin/brain.nn");
    cout << calculate_accuracy(brain) << endl;
    //learning_rate = pow((1.0 - calculate_accuracy(brain)),2.0);
    usleep(4 * 1000000);
    }
    delete[] input;
    delete[] target_output;
    return 0;
}
