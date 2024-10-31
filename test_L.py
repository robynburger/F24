import numpy as np
import matplotlib.pyplot as plt


'''
Usage: 

For testing case 1 or case 2, change penultimate line to vary your m value and 
ignore the rest of this preamble. 

For randomized strings: 
Enter values for string and verbose
then in command line:
$ python LTS.py

'''

#characters allowed in test string 
alphabet = ['a', 'b', 'c', 'd', 'e']

# max size of the test string 
max_length = 15

# number of test cases
num_tests = 100000

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
  L = [0]
  n = len(s) 
  for i in range(1, n):       
    L.append(F[n-1][i-1][i])
  L.append(0)
  
  return L
"""
Returns the number of peaks and their indices.
"""
def count_peaks(L):
    # peaks = 0
    peak_list = []
    n = len(L)
    
    for i in range(1, n - 1):
        # Check if there's an increasing sequence that turns into a decreasing one.
        if L[i] > L[i - 1]:
            j = i + 1
            while j < n and L[j] == L[i]:
                j += 1  # Skip over equal values
            if j < n and L[i] > L[j]:  # Verify it eventually decreases
                # peaks += 1
                peak_list.append(i)
                i = j - 1  # Move the pointer to skip checking these values again

    return peak_list
"""
Returns the valley depth, as defined in week 7 notes
"""
def valley_depth(L, p_list): 
  depth = 0
  if len(p_list) > 1:
    for i in range(0, len(p_list)-1):
      peak1 = p_list[i]
      peak2 = p_list[i+1]
      height1, height2 = L[peak1], L[peak2]
      d = abs(height1 - height2)
      if d > depth: depth = d
  return depth
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
  p = find_p(F, s) #optimal splitpoint 
  L = find_L(F, s)
  c = len(set(s))
  p_l = count_peaks(L) #peak list
  n_p = len(p_l) # number of peaks
  v = valley_depth(L, p_l) # True if peaks heights differ by more than 1 
  print(f"s: {s} \n L: {L} \n chars: {c} \n {n_p} peaks: {p_l} \n \n")
  return L
"""
Plots the graphs
"""
def plot(L, m, n):
    plt.figure()
    x = list(range(len(L)))
    y = L
    plt.plot(x, y, marker='o', color='blue', linewidth=2, markersize=5)
    for i, txt in enumerate(y):
      plt.text(x[i], y[i] + 0.1, str(txt), ha='center', va='bottom', fontsize=10)
    max_ticks = 10  
    x_ticks = np.linspace(0, len(L) - 1, min(len(L), max_ticks), dtype=int)
    y_ticks = np.linspace(min(y), max(y), max_ticks, dtype=int)
    plt.xticks(x_ticks) 
    plt.yticks(y_ticks) 
    plt.xlabel(f"{L}")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_position(('data', 0))  
    plt.title(f"m = {m}, n = {n}")
    plt.draw

"""
Generates the string of the form of case 1 or case 2
"""
def test_cases(m):
  for n in range(2, m):
    s = ""
    for _ in range(0, m): s += ('a')
    for _ in range(0, n): s += ('ba') # case 1: ab, case 2: ba
    for _ in range(0, m): s += ('b')
    print(f"m = {m}, n = {n}")
    plot(test_L(s), m, n)
  plt.show()


m = 7 #change this line 
test_cases(m)  




