import csv
import numpy as np


def format_data(data_set):
    data = []
    with open(data_set, "r") as f:
        dat = csv.reader(f)
        for row in dat:
            data.append(row)
    list_length = len(data)
    x = [data[i][0:2] for i in range(list_length)]
    t = [data[i][2] for i in range(list_length)]
    return x, t

def perceptron(x , w , theta):
        theta = np.array(theta,dtype='float')
        x = np.array(x,dtype='float')
        w = np.array(w,dtype='float')
        wx = np.dot(w,x)
        V = np.tanh(wx - theta)
        return V



def hidden_layer_neuron(n_epoch,x_values,targets,eta,M1,M2):
    targets = np.array(targets,dtype='float')
    x_values = np.array(x_values,dtype='float')

    g_prim = lambda b: 1 - np.tanh(b)**2

    w_1 = np.random.randn(M1,2) 
    w_2 = np.random.randn(M2,M1) 
    w_3 = np.random.randn(M2) 

    theta_1 = np.zeros(M1)
    theta_2 = np.zeros(M2)
    theta_3 = 0   

    for epoch in range(n_epoch):
        indices = np.random.permutation(len(x_values))
        for mu in indices:



            V1 = perceptron(x_values[mu],w_1,theta_1)
            V2 = perceptron(V1,w_2,theta_2)
            O = perceptron(V2,w_3,theta_3)

            output_error =  g_prim(np.dot(w_3,V2) - theta_3) * (targets[mu] - O)
            error_2 = np.dot(output_error, w_3) * g_prim(np.dot(w_2,V1) - theta_2)
            error_1 = np.dot(error_2, w_2) * g_prim(np.dot(w_1,x_values[mu]) - theta_1)

            w_1 += eta * np.outer(error_1,x_values[mu])
            w_2 += eta * np.outer(error_2,V1)
            w_3 += eta * output_error * V2

            theta_1 -= eta * error_1
            theta_2 -= eta * error_2
            theta_3 -= eta * output_error

        C = validation(
        x_validation,
        t_validation,
        w_1, w_2, w_3,
        theta_1, theta_2, theta_3
        )
        print(epoch, C)
     
    return w_1,w_2,w_3,theta_1,theta_2,theta_3




def validation(x_validation,t_validation,w_1,w_2,w_3,theta_1,theta_2,theta_3):
    t_validation = np.array(t_validation,dtype='float')
    x_validation = np.array(x_validation,dtype='float')
    p_val = len(x_validation)

    predicitons = []
    error_list = []

    for x, t in zip(x_validation, t_validation):
        V1 = perceptron(x,w_1,theta_1)
        V2 = perceptron(V1,w_2,theta_2)
        O = perceptron(V2,w_3,theta_3)
        O = 1 if O > 0 else -1
        error = abs(O - t)
        error_list.append(error)
        predicitons.append(O)
    
    C = 1 / (2*p_val) * sum(error_list)

    return C 

n_epoch = 100
eta     = 0.01
M1      = 10
M2      = 8

x_values, targets = format_data('training_set.csv')
x_validation, t_validation = format_data('validation_set.csv')  
w_1,w_2,w_3,theta_1,theta_2,theta_3 = hidden_layer_neuron(n_epoch,x_values,targets,eta,M1,M2)
percentage = validation(x_validation,t_validation,w_1,w_2,w_3,theta_1,theta_2,theta_3)
          








