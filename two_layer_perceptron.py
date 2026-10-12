import csv
import numpy as np


def format_data(data_set):
    data = []
    with open(data_set, "r") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            data.append(row)
    number_of_samples = len(data)
    input_values = [data[i][0:2] for i in range(number_of_samples)]
    targets = [data[i][2] for i in range(number_of_samples)]
    return input_values, targets

def two_layer_perceptron(input_values, weights, thresholds):
        thresholds = np.array(thresholds,dtype='float')
        input_values = np.array(input_values,dtype='float')
        weights = np.array(weights,dtype='float')
        weighted_input = np.dot(weights,input_values)
        activations = np.tanh(weighted_input - thresholds)
        return activations



def train_two_layer_perceptron(number_of_epochs,
                               training_inputs,
                               training_targets,
                               learning_rate,
                               hidden_layer_1_size,
                               hidden_layer_2_size):
    
    training_targets = np.array(training_targets,dtype='float')
    training_inputs = np.array(training_inputs,dtype='float')

    tanh_derivative = lambda b: 1 - np.tanh(b)**2

    weights_input_to_hidden_1 = np.random.randn(hidden_layer_1_size,2) 
    weights_hidden_1_to_hidden_2 = np.random.randn(hidden_layer_2_size,hidden_layer_1_size) 
    weights_hidden_2_to_output = np.random.randn(hidden_layer_2_size) 

    thresholds_hidden_1 = np.zeros(hidden_layer_1_size)
    thresholds_hidden_2 = np.zeros(hidden_layer_2_size)
    threshold_output = 0   

    for epoch_index in range(number_of_epochs):
        sample_order = np.random.permutation(len(training_inputs))
        for sample_index in sample_order:

            hidden_1_activations = two_layer_perceptron(training_inputs[sample_index],
                                                        weights_input_to_hidden_1,
                                                        thresholds_hidden_1)
            
            hidden_2_activations = two_layer_perceptron(hidden_1_activations,
                                                        weights_hidden_1_to_hidden_2,
                                                        thresholds_hidden_2)
            
            output  = two_layer_perceptron(hidden_2_activations,
                                           weights_hidden_2_to_output,
                                           threshold_output)

            output_delta =  tanh_derivative(np.dot(weights_hidden_2_to_output,hidden_2_activations) - threshold_output) * (training_targets[sample_index] - output)
            hidden_2_delta = np.dot(output_delta, weights_hidden_2_to_output) * tanh_derivative(np.dot(weights_hidden_1_to_hidden_2,hidden_1_activations) - thresholds_hidden_2)
            hidden_1_delta = np.dot(hidden_2_delta, weights_hidden_1_to_hidden_2) * tanh_derivative(np.dot(weights_input_to_hidden_1,training_inputs[sample_index]) - thresholds_hidden_1)

            weights_input_to_hidden_1 += learning_rate * np.outer(hidden_1_delta,training_inputs[sample_index])
            weights_hidden_1_to_hidden_2 += learning_rate * np.outer(hidden_2_delta,hidden_1_activations)
            weights_hidden_2_to_output += learning_rate * output_delta * hidden_2_activations

            thresholds_hidden_1 -= learning_rate * hidden_1_delta
            thresholds_hidden_2 -= learning_rate * hidden_2_delta
            threshold_output -= learning_rate * output_delta

        validation_error = validation(
        validation_inputs,
        validation_targets,
        weights_input_to_hidden_1, weights_hidden_1_to_hidden_2, weights_hidden_2_to_output,
        thresholds_hidden_1, thresholds_hidden_2, threshold_output
        )
        print(epoch_index, validation_error)

    return weights_input_to_hidden_1,weights_hidden_1_to_hidden_2,weights_hidden_2_to_output,thresholds_hidden_1,thresholds_hidden_2,threshold_output




def validation(validation_inputs,
               validation_targets,
               weights_input_to_hidden_1,
               weights_hidden_1_to_hidden_2,
               weights_hidden_2_to_output,
               thresholds_hidden_1,thresholds_hidden_2,
               threshold_output):
    
    validation_targets = np.array(validation_targets,dtype='float')
    validation_inputs = np.array(validation_inputs,dtype='float')
    number_of_samples = len(validation_inputs)

    predictions = []
    sample_errors = []

    for input_sample, target in zip(validation_inputs, validation_targets):
        hidden_1_activations = two_layer_perceptron(input_sample,weights_input_to_hidden_1,thresholds_hidden_1)
        hidden_2_activations = two_layer_perceptron(hidden_1_activations,weights_hidden_1_to_hidden_2,thresholds_hidden_2)
        output = two_layer_perceptron(hidden_2_activations,weights_hidden_2_to_output,threshold_output)
        output = 1 if output > 0 else -1
        sample_error = abs(output - target)
        sample_errors.append(sample_error)
        predictions.append(output)
    
    validation_error = 1 / (2*number_of_samples) * sum(sample_errors)

    return validation_error 

number_of_epochs = 100
learning_rate     = 0.01
hidden_layer_1_size     = 10
hidden_layer_2_size      = 8

training_inputs, training_targets = format_data('training_set.csv')
validation_inputs, validation_targets = format_data('validation_set.csv')  
weights_input_to_hidden_1, weights_hidden_1_to_hidden_2 ,weights_hidden_2_to_output ,threshold_hidden_1 ,thresholds_hidden_2 ,threshold_output = train_two_layer_perceptron(number_of_epochs,training_inputs,training_targets,learning_rate,hidden_layer_1_size,hidden_layer_2_size)
validation_error = validation(validation_inputs,
                              validation_targets,
                              weights_input_to_hidden_1,
                              weights_hidden_1_to_hidden_2,
                              weights_hidden_2_to_output,
                              threshold_hidden_1,
                              thresholds_hidden_2,
                              threshold_output)
          








