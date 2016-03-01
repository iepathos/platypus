#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DeepMind's NeuralStack implemented with Python 3
# Followed tutorial here: https://iamtrask.github.io/2016/02/25/deepminds-neural-stack-machine/?i=4
# and made it Python 3 compatible.
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_out2deriv(out):
    return out * (1 - out)


def tanh(x):
    return np.tanh(x)


def tanh_out2deriv(out):
    return (1 - out**2)


def relu(x, deriv=False):
    if(deriv):
        return int(x >= 0)
    return max(0, x)


class NeuralStack():

    def __init__(self, stack_width=2, o_prime_dim=6):
        self.stack_width = stack_width
        self.o_prime_dim = o_prime_dim
        self.reset()

    def reset(self):
        # INIT STACK
        self.V = list()  # stack states
        self.s = list()  # stack strengths
        self.d = list()  # push strengths
        self.u = list()  # pop strengths
        self.r = list()
        self.o = list()

        self.V_delta = list()  # stack states
        self.s_delta = list()  # stack strengths
        self.d_error = list()  # push strengths
        self.u_error = list()  # pop strengths

        self.t = 0

    def pushAndPopForward(self, v_t, d_t, u_t):

        self.d.append(d_t)
        self.d_error.append(0)

        self.u.append(u_t)
        self.u_error.append(0)

        new_s = np.zeros(self.t+1)
        for i in range(self.t+1):
            new_s[i] = self.s_t(i)
        self.s.append(new_s)
        self.s_delta.append(np.zeros_like(new_s))

        if(len(self.V) == 0):
            V_t = np.zeros((0, self.stack_width))
        else:
            V_t = self.V[-1]
        self.V.append(np.concatenate((V_t, np.atleast_2d(v_t)), axis=0))
        self.V_delta.append(np.zeros_like(self.V[-1]))

        r_t = self.r_t()
        self.r.append(r_t)

        self.t += 1
        return r_t

    def s_t(self, i):
        if(i >= 0 and i < self.t):
            inner_sum = self.s[self.t-1][i+1:self.t-0]
            return relu(self.s[self.t-1][i] - relu(self.u[self.t] - np.sum(inner_sum)))
        elif(i == self.t):
            return self.d[self.t]
        else:
            print("Problem")

    def s_t_error(self, i, error):
        if(i >= 0 and i < self.t):
            if(self.s_t(i) >= 0):
                self.s_delta[self.t-1][i] += error
                if(relu(self.u[self.t] - np.sum(self.s[self.t-1][i+1:self.t-0])) >= 0):
                    self.u_error[self.t] -= error
                    self.s_delta[self.t-1][i+1:self.t-0] += error
        elif(i == self.t):
            self.d_error[self.t] += error
        else:
            print("Problem")

    def r_t(self):
        r_t_out = np.zeros(self.stack_width)
        for i in range(0, self.t+1):
            temp = min(self.s[self.t][i],relu(1 - np.sum(self.s[self.t][i+1:self.t+1])))
            r_t_out += temp * self.V[self.t][i]
        return r_t_out

    def r_t_error(self, r_t_error):
        for i in range(0, self.t+1):
            temp = min(self.s[self.t][i],relu(1 - np.sum(self.s[self.t][i+1:self.t+1])))
            self.V_delta[self.t][i] += temp * r_t_error
            temp_error = np.sum(r_t_error * self.V[self.t][i])

            if(self.s[self.t][i] < relu(1 - np.sum(self.s[self.t][i+1:self.t+1]))):
                self.s_delta[self.t][i] += temp_error
            else:
                if(relu(1 - np.sum(self.s[self.t][i+1:self.t+1])) > 0):
                    self.s_delta[self.t][i+1:self.t+1] -= temp_error  # minus equal becuase of the (1-).. and drop the 1

    def backprop_single(self, r_t_error):
        self.t -= 1
        self.r_t_error(r_t_error)
        for i in reversed(range(self.t+1)):
            self.s_t_error(i, self.s_delta[self.t][i])

    def backprop(self, all_errors_in_order_of_training_data):
        errors = all_errors_in_order_of_training_data
        for error in reversed(list((errors))):
            self.backprop_single(error)


options = 2
sub_sequence_length = 5
sequence_length = sub_sequence_length*2

sequence = (np.random.random(sub_sequence_length)*options).astype(int)+2
sequence

X = np.zeros((sub_sequence_length*2, options+2))
Y = np.zeros_like(X)
for i in range(len(sequence)):
    X[i][sequence[i]] = 1
    X[-i-1][0] = 1
    X[i][1] = 1
    Y[-i-1][sequence[i]] = 1

sequence_length = len(X)
x_dim = X.shape[1]
h_dim = 8
o_prime_dim = 8
stack_width = 2
y_dim = Y.shape[1]


np.random.seed(1)
sub_sequence_length = 2

W_xh = (np.random.rand(x_dim, h_dim)*0.2) - 0.1
W_xh_update = np.zeros_like(W_xh)

W_hox = (np.random.rand(h_dim, x_dim)*0.2) - 0.1
W_hox_update = np.zeros_like(W_hox)

W_opx = (np.random.rand(o_prime_dim, x_dim)*0.2) - 0.1
W_opx_update = np.zeros_like(W_opx)

W_hh = (np.random.rand(h_dim, h_dim)*0.2) - 0.1
W_hh_update = np.zeros_like(W_hh)

W_rh = (np.random.rand(stack_width, h_dim)*0.2) - 0.1
W_rh_update = np.zeros_like(W_rh)

b_h = (np.random.rand(h_dim)*0.2) - 0.1
b_h_update = np.zeros_like(b_h)

W_hop = (np.random.rand(h_dim, o_prime_dim) * 0.2) - 0.1
W_hop_update = np.zeros_like(W_hop)

b_op = (np.random.rand(o_prime_dim)*0.2) - 0.1
b_op_update = np.zeros_like(b_op)

W_op_d = (np.random.rand(o_prime_dim, 1)*0.2) - 0.1
W_op_d_update = np.zeros_like(W_op_d)

W_op_u = (np.random.rand(o_prime_dim, 1)*0.2) - 0.1
W_op_u_update = np.zeros_like(W_op_u)

W_op_v = (np.random.rand(o_prime_dim, stack_width)*0.2) - 0.1
W_op_v_update = np.zeros_like(W_op_v)

W_op_o = (np.random.rand(o_prime_dim, y_dim)*0.2) - 0.1
W_op_o_update = np.zeros_like(W_op_o)

b_d = (np.random.rand(1)*0.2)+1
b_d_update = np.zeros_like(b_d)

b_u = (np.random.rand(1)*0.2)-1
b_u_update = np.zeros_like(b_u)

b_v = (np.random.rand(stack_width)*0.2)-0.1
b_v_update = np.zeros_like(b_v)

b_o = (np.random.rand(y_dim)*0.2)-0.1
b_o_update = np.zeros_like(b_o)

error = 0
reconstruct_error = 0
reconstruct_error_2 = 0
max_len = 3
batch_size = 50
for it in range(750000):

#     if(it % 100 == 0):
    sub_sequence_length = np.random.randint(max_len)+3
    sequence = (np.random.random(sub_sequence_length)*options).astype(int)+2
    sequence

    X = np.zeros((sub_sequence_length*2, options+2))
    Y = np.zeros_like(X)
    for i in range(len(sequence)):
        X[i][sequence[i]] = 1
        X[-i-1][0] = 1
        X[i][1] = 1
        Y[-i-1][sequence[i]] = 1


    layers = list()
    stack = NeuralStack(stack_width=stack_width,o_prime_dim=o_prime_dim)    
    for i in range(len(X)):
        layer = {}

        layer['x'] = X[i]

        if(i == 0):
            layer['h_t-1'] = np.zeros(h_dim)
#             layer['h_t-1'][0] = 1
            layer['r_t-1'] = np.zeros(stack_width)
#             layer['r_t-1'][0] = 1
        else:
            layer['h_t-1'] = layers[i-1]['h_t']
            layer['r_t-1'] = layers[i-1]['r_t']

        layer['h_t'] = tanh(np.dot(layer['x'],W_xh) + np.dot(layer['h_t-1'],W_hh) + np.dot(layer['r_t-1'],W_rh) + b_h)
        layer['xo_t'] = sigmoid(np.dot(layer['h_t'],W_hox))
        layer['o_prime_t'] = tanh(np.dot(layer['h_t'],W_hop)+b_op)
        layer['o_prime_x_t'] = sigmoid(np.dot(layer['o_prime_t'],W_opx))
        layer['o_t'] = sigmoid(np.dot(layer['o_prime_t'],W_op_o) + b_o)

        if(i < len(X)-1):
            layer['d_t'] = sigmoid(np.dot(layer['o_prime_t'],W_op_d) + b_d)
            layer['u_t'] = sigmoid(np.dot(layer['o_prime_t'],W_op_u) + b_u)
            layer['v_t'] = tanh(np.dot(layer['o_prime_t'],W_op_v) + b_v)

            layer['r_t'] = stack.pushAndPopForward(layer['v_t'],layer['d_t'],layer['u_t'])

        layers.append(layer)

    for i in list(reversed(range(len(X)))):
        layer = layers[i]

        layer['o_t_error'] = (Y[i] - layer['o_t'])

        if(i>0):
            layer['xo_t_error'] = layers[i-1]['x'] - layer['xo_t']
            layer['xo_t_delta'] = layer['xo_t_error'] * sigmoid_out2deriv(layer['xo_t'])            
            
            layer['x_o_prime_x_t_error'] = (layers[i-1]['x'] - layer['o_prime_x_t'])
            layer['x_o_prime_x_t_delta'] = layer['x_o_prime_x_t_error'] * sigmoid_out2deriv(layer['o_prime_x_t'])
        else:
            layer['xo_t_delta'] = np.zeros_like(layer['x'])
            layer['x_o_prime_x_t_delta'] = np.zeros_like(layer['x'])
#         if(it > 2000):
        layer['xo_t_delta'] *= 1
        layer['x_o_prime_x_t_delta'] *= 1
        

        error += np.sum(np.abs(layer['o_t_error']))
        if(i > 0):
            reconstruct_error += np.sum(np.abs(layer['xo_t_error']))
            reconstruct_error_2 += np.sum(np.abs(layer['x_o_prime_x_t_error']))
        if(it % 100 == 99):
            if(i == len(X)-1):
    
                if(it % 1000 == 999):
                    print("MaxLen:"+str(max_len)+ " Iter:" + str(it) + " Error:" + str(error)+ " RecError:" + str(reconstruct_error) + " RecError2:"+ str(reconstruct_error_2) + " True:" + str(sequence) + " Pred:" + str(list(map(lambda x:np.argmax(x['o_t']),layers[sub_sequence_length:]))))
                    if(it % 10000 == 9999):
                        print("U:" + str(np.array(stack.u).T[0]))
                        print("D:" + str(np.array(stack.d).T[0]))
#                     print "o_t:"
#                     for l in layers[sub_sequence_length:]:
#                         print l['o_t'] 
#                     print "V_t:"
#                     for row in stack.V[-1]:
#                         print row
                if(error < max_len+4 and it > 10000):
                    max_len += 1
                    it = 0
                error = 0
                reconstruct_error = 0
                reconstruct_error_2 = 0

        layer['o_t_delta'] = layer['o_t_error'] * sigmoid_out2deriv(layer['o_t'])

        layer['o_prime_t_error'] = np.dot(layer['o_t_delta'],W_op_o.T)
        layer['o_prime_t_error'] += np.dot(layer['x_o_prime_x_t_delta'],W_opx.T)
        if(i < len(X)-1):
            layer['r_t_error'] = layers[i+1]['r_t-1_error']
            stack.backprop_single(layer['r_t_error'])

            layer['v_t_error'] = stack.V_delta[i][i]
            layer['v_t_delta'] = layer['v_t_error'] * tanh_out2deriv(layer['v_t'])
            layer['o_prime_t_error'] += np.dot(layer['v_t_delta'],W_op_v.T)

            layer['u_t_error'] = stack.u_error[i]
            layer['u_t_delta'] = layer['u_t_error'] * sigmoid_out2deriv(layer['u_t'])
            layer['o_prime_t_error'] += np.dot(layer['u_t_delta'],W_op_u.T)

            layer['d_t_error'] = stack.d_error[i]
            layer['d_t_delta'] = layer['d_t_error'] * sigmoid_out2deriv(layer['d_t'])
            layer['o_prime_t_error'] += np.dot(layer['d_t_delta'],W_op_d.T)


        layer['o_prime_t_delta'] = layer['o_prime_t_error'] * tanh_out2deriv(layer['o_prime_t'])
        layer['h_t_error'] = np.dot(layer['o_prime_t_delta'],W_hop.T)
        layer['h_t_error'] += np.dot(layer['xo_t_delta'],W_hox.T)
        if(i < len(X)-1):
            layer['h_t_error'] += layers[i+1]['h_t-1_error']

        layer['h_t_delta'] = layer['h_t_error'] * tanh_out2deriv(layer['h_t'])
        layer['h_t-1_error'] = np.dot(layer['h_t_delta'],W_hh.T)
        layer['r_t-1_error'] = np.dot(layer['h_t_delta'],W_rh.T)

    for i in range(len(X)):
        layer = layers[i]
        if(it<2000):
            max_alpha = 0.05 * batch_size
#         else:
#             max_alpha = 0.05 * batch_size
        alpha = max_alpha / sub_sequence_length

        W_xh_update += alpha * np.outer(layer['x'],layer['h_t_delta'])
        W_hh_update += alpha * np.outer(layer['h_t-1'],layer['h_t_delta'])
        W_rh_update += alpha * np.outer(layer['r_t-1'],layer['h_t_delta'])
        W_hox_update += alpha * np.outer(layer['h_t'],layer['xo_t_delta'])
        
        b_h_update += alpha * layer['h_t_delta']

        W_hop_update += alpha * np.outer(layer['h_t'],layer['o_prime_t_delta'])
        b_op_update += alpha * layer['o_prime_t_delta']
        
        W_opx_update += alpha * np.outer(layer['o_prime_t'],layer['x_o_prime_x_t_delta'])
        
        if(i < len(X)-1):
            W_op_d_update += alpha * np.outer(layer['o_prime_t'],layer['d_t_delta'])
            W_op_u_update += alpha * np.outer(layer['o_prime_t'],layer['u_t_delta'])
            W_op_v_update += alpha * np.outer(layer['o_prime_t'],layer['v_t_delta'])

            b_d_update += alpha * layer['d_t_delta']# * 10
            b_u_update += alpha * layer['u_t_delta']# * 10
            b_v_update += alpha * layer['v_t_delta']

        W_op_o_update += alpha * np.outer(layer['o_prime_t'],layer['o_t_delta'])
        b_o_update += alpha * layer['o_t_delta']


    if(it % batch_size == (batch_size-1)):
        W_xh += W_xh_update/batch_size
        W_xh_update *= 0
        
        W_hh += W_hh_update/batch_size
        W_hh_update *= 0
        
        W_rh += W_rh_update/batch_size
        W_rh_update *= 0
        
        b_h += b_h_update/batch_size
        b_h_update *= 0
        
        W_hop += W_hop_update/batch_size
        W_hop_update *= 0
        
        b_op += b_op_update/batch_size
        b_op_update *= 0
        
        W_op_d += W_op_d_update/batch_size
        W_op_d_update *= 0
        
        W_op_u += W_op_u_update/batch_size
        W_op_u_update *= 0
        
        W_op_v += W_op_v_update/batch_size
        W_op_v_update *= 0
        
        W_opx += W_opx_update/batch_size
        W_opx_update *= 0
        
        W_hox += W_hox_update/batch_size
        W_hox_update *= 0
        
        b_d += b_d_update/batch_size
        b_d_update *= 0
        
        b_u += b_u_update/batch_size
        b_u_update *= 0
        
        b_v += b_v_update/batch_size
        b_v_update *= 0
        
        W_op_o += W_op_o_update/batch_size
        W_op_o_update *= 0
        
        b_o += b_o_update/batch_size
        b_o_update *= 0
