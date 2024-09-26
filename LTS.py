# import os
import numpy as np

'''
Usage: 
Change "string" below, then in command line:
$ python LTS.py

'''
string = "BABBCA"


'''
Populates the nxn matrix, Fk, for each 0 < k <= n such that F[k][i][j] = F_k(i, j)
'''
def populate_F(F, s):
  n = len(s)
  for k in range(1, n+1):
    for i in range (1, k+1): 
      for j in range(i+1, k+1):
            # print(f"\nk: {k}. i: {i}, j: {j}")
            if s[i-1] == s[k-1]: 
              # print(f"i-1 = {i-1}, s[i-1] = {s[i-1]}")
              # print(f"k-1 = {k-1}, s[i-1] = {s[i-1]}")
              F[k][i][j] = F[k-1][i-1][j] + 1 
              # print(f'F[{k}][{i}][{j}] = {F[k][i][j]}')
            else:
              # print(f'F[{k}][{i-1}][{j}] = {F[k][i-1][j]}')
              # print(f'F[{k-1}][{i}][{j}] = {F[k-1][i][j]}')
              # print(f"\t F[{k}][{i}][{j}] = {F[k][i][j]}")
              F[k][i][j]= max(F[k][i-1][j],F[k-1][i][j])
  return F

'''
Populates the nxn matrix, Dk, for each 0 < k <= n such that D[k][i][j] = D_k(i, j)
'''
def populate_D(F, n):
  D = np.zeros((n, n, n), dtype=int)
  for k in range(1, n+1):
    for i in range (1, n+1):
     for j in range(i+1, n+1):
        D[k-1][i-1][j-1] = F[k][i][j] - F[k][i-1][j]
  return D

'''
Populates the nxn matrix A where A[i][k] = a_k(i)
'''
def populate_A(D, n):
  A = np.zeros((n, n), dtype=int)
  for k in range(1, n+1):
    for i in range(1, k+1):
        for j in range(i+1, k+1): 
          if D[k-1][i-1][j-1] == 1:
            A[i-1][k-1] = j
  return A
      
'''
Prints F, D, and A for input string 's'
'''
def LTS(s):
  n = len(s)
  # Pyton allows you to access the -1^st index, so initially, the F matrices have a buffer row and column of 0s
  empty_bufferF = np.zeros((n+1, n+1, n+1), dtype=int)
  bufferF = populate_F(empty_bufferF, s)

  D = populate_D(bufferF, n)
  
  # remove F's buffer row/column 
  F = np.zeros((n, n, n), dtype=int)
  for k in range(1, n+1):
    for i in range (1, n+1):
     for j in range(1, n+1):
        F[k-1][i-1][j-1] = bufferF[k][i][j]
    print(f"F{k}: \n {F[k-1]}")
    print(f"D{k}: \n {D[k-1]} \n")
  A = populate_A(D, n)
  print(f"A: \n {A}")
  return 0
  
LTS(string)
