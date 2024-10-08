import random
import numpy as np

'''
Usage: 
Enter values for string and verbose
then in command line:
$ python LTS.py

'''

#characters allowed in test string 
alphabet = ['a', 'b', 'c', 'd']

# max size of the test string 
max_length = 10

# number of test cases
num_tests = 10000000

# The input string 
string = "aaba" 

# if True, output includes F and D matrices for all 0<k<n+1
# if False, output only A matrix
verbose = False 

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

def count_peaks(L):
    peaks = 0
    n = L.size
    
    for i in range(1, n - 1):
        # Check if there's an increasing sequence that turns into a decreasing one.
        if L[0][i] > L[0][i - 1]:
            j = i + 1
            while j < n and L[0][j] == L[0][i]:
                j += 1  # Skip over equal values
            if j < n and L[0][i] > L[0][j]:  # Verify it eventually decreases
                peaks += 1
                i = j - 1  # Move the pointer to skip checking these values again

    return peaks
    
'''
Prints F, D, and A for input string 's'
'''
def test_L(s):
  n = len(s)
  # Pyton allows you to access the -1^st index, so initially, the F matrices have a buffer row and column of 0s
  empty_bufferF = np.zeros((n+1, n+1, n+1), dtype=int)
  bufferF = populate_F(empty_bufferF, s)
  
  # remove F's buffer row/column 
  F = np.zeros((n, n, n), dtype=int)
  for k in range(1, n+1):
    for i in range (1, n+1):
     for j in range(1, n+1):
        F[k-1][i-1][j-1] = bufferF[k][i][j]
      # print(f"F{k}: \n {F[k-1]}")
  p = find_p(F, s)
  L = find_L(F, s)
  # print(f"len = {L.size}")
  c = len(set(s))
  p = count_peaks(L)
  print(f"s: {s} \n L: {L} \n chars: {c} \n peaks : {p}\n \n")
  return c < p 

not_failed = True
for x in range(num_tests):
    if not_failed:
        s = ""
        for _ in range(random.randint(max_length-5, max_length)):
            s += str(random.choice(alphabet))
        #  s = "aabbcc"
        if test_L(s):
          not_failed = False
          print("!!!!")

