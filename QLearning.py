# -*- coding: utf-8 -*-
import numpy as np

def imprime(Q):
  rows = len(Q); cols = len(Q[0])
  for i in range(rows):
    print("%d " % i, end="")
    if i < 10: 
        print(" ", end="")
    for j in range(cols): 
        print(" %6.2f" % Q[i,j], end="")
    print("")
  print("")

def prox_estados(s, F, ns):
  prox_estados = []
  for j in range(ns):
    if F[s,j] == 1: 
        prox_estados.append(j)
  return prox_estados

def prox_estado_randomico(s, F, ns):
  proximos_estados_possiveis = prox_estados(s, F, ns)
  prox_estado = proximos_estados_possiveis[np.random.randint(0, len(proximos_estados_possiveis))]
  return prox_estado 


def QL(F, R, Q, gamma, lrn_rate, goal, ns, max_epochs):
  for i in range(0,max_epochs):
    curr_s = np.random.randint(0,ns)

    while(True):
      next_s = prox_estado_randomico(curr_s, F, ns)
      poss_next_next_states = prox_estados(next_s, F, ns)

      max_Q = -9999.99
      for j in range(len(poss_next_next_states)):
        nn_s = poss_next_next_states[j]
        q = Q[next_s,nn_s]
        if q > max_Q:
          max_Q = q
      # Q = [(1-a) * Q]  +  [a * (rt + (g * maxQ))]
      Q[curr_s][next_s] = ((1 - lrn_rate) * Q[curr_s] [next_s]) + (lrn_rate * (R[curr_s][next_s] + (gamma * max_Q)))
      curr_s = next_s
      
      if curr_s == goal: 
          break

def main():
  np.random.seed(1)
  
  F = np.zeros(shape=[3,2], dtype=np.int)
  F[0,1] = 1; F[1,0] = 1;

  R = np.zeros(shape=[15,15], dtype=np.int)
  R[0,1] = -0.1; R[1,0] = -0.1;

  Q = np.zeros(shape=[3,2], dtype=np.float32)
  
  print("Q-learning")
  
  goal = 1
  ns = 2
  gamma = 0.9
  lrn_rate = 0.1
  max_epochs = 1000
  QL(F, R, Q, gamma, lrn_rate, goal, ns, max_epochs)
  
  print(Q)
  
  print("A matriz Q é: \n ")
  imprime(Q)


main()
