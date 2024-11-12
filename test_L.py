import random
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
alphabet = ['a', 'b']

# max size of the test string 
max_length = 20

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
  # print(f"s: {s} \n L: {L} \n chars: {c} \n {n_p} peaks: {p_l} \n \n")
  return L
"""
Plots the graphs
"""
def plot(s, L, m, n):
    plt.figure()
    x = list(range(len(L)))
    y = L
    plt.plot(x, y, marker='o', color='blue', linewidth=2, markersize=5)
    
    max_ticks = 10
    x_ticks = np.linspace(0, len(L) - 1, min(len(L), max_ticks), dtype=int)
    y_ticks = np.linspace(min(y), max(y), len(x_ticks), dtype=int)  # Match the number of y-ticks to x-ticks for visual proportion
    
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    
    plt.xlabel(f"{L}")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_position(('data', 0)) 
    plt.gca().spines['bottom'].set_position(('data', 0))

    # Set tight limits on both axes
    plt.xlim(0, len(L) - 1)
    plt.ylim(min(y), max(y) + 0.3)  # Add minimal padding to make y-axis proportional without excessive spacing
    plt.gca().set_box_aspect(.25)
    plt.title(f"s = {s},")
    plt.draw()
"""
Generates the string of the form of case 1 or case 2
"""
def test_case1(m):
  for n in range(1, m+1):
    s = ""
    for _ in range(0, m): s += ('a')
    for _ in range(0, n): s += ('ab') 
    for _ in range(0, m): s += ('b')
    print(f"m = {m}, n = {n}")
    plot(s, test_L(s), m, n)
  plt.show()
def test_case2(m):
  for n in range(1, m+1):
    s = ""
    for _ in range(0, m): s += ('a')
    for _ in range(0, n): s += ('ba') 
    for _ in range(0, m): s += ('b')
    print(f"m = {m}, n = {n}")
    plot(s, test_L(s), m, n)
  plt.show()
def test_case3(m, n):
  # for n in range(1, m+1):
    s = ""
    for _ in range(0, m): s += ('a')
    for _ in range(0, n): s += ('b')
    for _ in range(0, m): s += ('a')
    print(f"m = {m}, n = {n}")
    plot(s, test_L(s), m, n)
    plt.show()              

def count_ones(L):
  count = 0
  for el in L:
    if el == 1:
      count += 1
  return count

def test_rand(alphabet, max_length):
  for x in range(num_tests):
      if x % 1000 == 0: print(f"Test {x}")
      s = ""
      for _ in range(random.randint(max_length-5, max_length)):
        s += str(random.choice(alphabet))
      analyze(s)

def analyze(s):
  L = test_L(s)
  c = len(set(s)) # number of characters
  p_l = count_peaks(L) # peak list
  n_p = len(p_l) # number of peaks
  # if count_ones(L) > 5: # equivalent to, if there are 4+ internal ones
  print(f"s: {s} \n L: {L} \n chars: {c} \n {n_p} peaks: {p_l} \n \n")
 
# Function to generate all binary strings 
def genAllBinaryStrings(n, arr, i): 
    if i == n:
        analyze(''.join(str(x) for x in arr))
        return
    arr[i] = 0
    genAllBinaryStrings(n, arr, i + 1)  
    arr[i] = 1
    genAllBinaryStrings(n, arr, i + 1) 
 
n = 4 
genAllBinaryStrings(n, [None] * n, 0)

# s = "bbaaab"
# plot(s, test_L(s), 0, 0)
# plt.show()
# m = 7 #change this line 
# test_cases(m)  
# test_case3(5, 11)




