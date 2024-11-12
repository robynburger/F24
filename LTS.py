# import os
import numpy as np

'''
Usage: 
Enter values for string and verbose
then in command line:
$ python LTS.py

'''
# The input string 
string = "abaabcdc" 

# if True, output includes F and D matrices for all 0<k<n+1
# if False, output only A matrix
verbose = True 


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
Finds p, the optimal splitpoint such that fn(p, p+1) = max l : 1 <= l < n(fn(l, r+l)))
'''
def find_p(F, s):
  n = len(s)
  maxVal = 0
  p = -1
  for i in range(1, n):
    curr = F[n-1][i-1][i] 
    if curr > maxVal:  ## Not sure if this should be > or >=
      maxVal = curr 
      p = i
      # print(f"F[{n-1}][{i-1}][{i}]={curr}")
  return p
"""
Finds F_n(i, i+1) for each i 

"""
def find_L(F, s): 
  n = len(s)
  L = np.zeros((1, n+1), dtype = int)
  n = len(s) 
  for i in range(1, n):       
    L[0][i] = F[n-1][i-1][i] #append 0 to end when print
  
  return L

    
'''
Prints F, D, and A for input string 's'
'''
def LTS(s, verbose):
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
    if verbose:
      print(f"F{k}: \n {F[k-1]}")
      print(f"D{k}: \n {D[k-1]} \n")
  A = populate_A(D, n)
  p = find_p(F, s)
  print(f"A: \n {A}")
  print(f"p = {p}")
  print(f"L = {find_L(F, s)}")
  return 0

  
LTS(string, verbose)
