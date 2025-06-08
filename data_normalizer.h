#ifndef DATA_NORMALIZER
#define DATA_NORMALIZER

#include<iostream>
#include<cmath>
using namespace std;

class Data_normalizer{

protected:
   double normalization_mean = 0;
   double normalization_standard_deviation = 0;
   double normalization_maximum_input = 0;
   double normalization_minimum_input = 0;
   bool do_normalization = false;
   bool do_round_normalization = false;
   bool is_normalization_variables_different_scale = false;
   double* normalization_maximum_inputs;
   double* normalization_minimum_inputs;

public:
   void set_input_normalization_parameters(double mean,double standard_deviation, double minimum_input, double maximum_input,bool do_round){
	   normalization_mean = mean;
	   normalization_standard_deviation = standard_deviation;
	   normalization_minimum_input = minimum_input;
	   normalization_maximum_input = maximum_input;
	   do_round_normalization = do_round;
	   do_normalization = true;
   }

   void set_input_normalization_parameters(double mean,double standard_deviation,int number_of_inputs, double* minimum_input, double* maximum_input, bool do_round){
           normalization_mean = mean;
           normalization_standard_deviation = standard_deviation;
           normalization_minimum_inputs = new double[number_of_inputs];
           normalization_maximum_inputs = new double[number_of_inputs];
	   for(int i = 0; i < number_of_inputs;i++){
		   normalization_minimum_inputs[i] = minimum_input[i];
		   normalization_maximum_inputs[i] = maximum_input[i];
	   }
           do_round_normalization = do_round;
           do_normalization = true;
	   is_normalization_variables_different_scale = true;
   }


   void normalize_input_data(double* input,int number_of_inputs){
	if(!is_normalization_variables_different_scale){
	   if((normalization_mean == 0.5) && (normalization_standard_deviation == 0.5))
		   for(int i = 0; i < number_of_inputs;i++)
			   input[i] = (input[i] - normalization_minimum_input) / (normalization_maximum_input - normalization_minimum_input);
	   else
	   for(int i = 0; i < number_of_inputs;i++){
		   //squish the data to be be between 0 and 1 which makes
		   //the distribution's mean 0.5 and its standard deviation 0.5
		   //-0.5: make the mean from 0.5 to 0
		   //*2: make the standard deviation from 0.5 to 1
		   double x = ((input[i] - normalization_minimum_input) / (normalization_maximum_input - normalization_minimum_input) - 0.5) * 2;
		   if(do_round_normalization)
			   input[i] = round((x + normalization_mean) * (1.0 / normalization_standard_deviation));
		   else
		   input[i] = (x + normalization_mean) * (1.0 / normalization_standard_deviation);
	   }
	}
	else{
	   if((normalization_mean == 0.5) && (normalization_standard_deviation == 0.5))
                   for(int i = 0; i < number_of_inputs;i++)
                           input[i] = (input[i] - normalization_minimum_inputs[i]) / (normalization_maximum_inputs[i] - normalization_minimum_inputs[i]);
           else
           for(int i = 0; i < number_of_inputs;i++){
                   //squish the data to be be between 0 and 1 which makes
                   //the distribution's mean 0.5 and its standard deviation 0.5
                   //-0.5: make the mean from 0.5 to 0
                   //*2: make the standard deviation from 0.5 to 1
                   double x = ((input[i] - normalization_minimum_inputs[i]) / (normalization_maximum_inputs[i] - normalization_minimum_inputs[i]) - 0.5) * 2;
                   if(do_round_normalization)
                           input[i] = round((x + normalization_mean) * (1.0 / normalization_standard_deviation));
		   else
                   input[i] = (x + normalization_mean) * (1.0 / normalization_standard_deviation);
           }
	}
   }

};

#endif
