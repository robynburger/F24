import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import math


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
alphabet = 2

# max size of the test string 
max_length = 20

# number of test cases
num_tests = 100000

# if True, output includes F and D matrices for all 0<k<n+1
# if False, output only A matrix
verbose = False 

lst3 = []
lst4 = []
lst5 = []
lst = []
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
def plot(s, L, m):
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
    # plt.title(f"{m}")
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
    # print(f"m = {m}, n = {n}")
    # plot(s, test_L(s), m, n)
    # plt.show()    
    return s          
def test_case4(h): 
   s = ""
   for _ in range(0, h+1): s += ('0')
   for _ in range(0, 2*h+1): s += ('1')
   for _ in range(0, h): s += ('0')
   return s
def test_case5(i, k): 
   s = ""
   for _ in range(0, i+1): s += ('0')
   for _ in range(0, i+k+1): s += ('1')
   for _ in range(0, 2): s+= ('0')
   for _ in range(0, i-k+1): s += ('1')
   for _ in range(0, i+ 1): s += ('0')
   return s

def count_n(L, n):
  count = 0
  for el in L:
    if el == n:
      count += 1
  return count
def count_consecutive_ones(L):
    max_ones = 0
    current_ones = 0
    for el in L:
        if el == 1:
            current_ones += 1
        else:
            max_ones = max(max_ones, current_ones)
            current_ones = 0
    return max(max_ones, current_ones)
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
  c1 = count_n(L, 1)
  return L
    # print(f"s: {s} \n L: {L} \n")
    # lst3.append(s)
  # if L == [0, 1, 1, 2, 1, 1, 0]:
  #   plot(s, test_L(s), 0, 0)
  #   plt.show()
  # if c1 == 4:
  #   lst4.append(s)
  # if c1 == 5:
  #   lst5.append(s)
  # if c1 > 5:
  #    lst.append(s)
 
 
def genBinary(n):
    return [''.join(p) for p in product('01', repeat=n)]
def genTernary(n):
    return [''.join(p) for p in product('012', repeat=n)]
def gen4ary(n):
    return [''.join(p) for p in product('0123', repeat=n)]
def gen5ary(n):
    return [''.join(p) for p in product('01234', repeat=n)]

def count_to_h(h, L):
  count = 0
  for i in L:
    if i < h+1:
      count = count + 1
  return count

# returns splitpoint(s) that result(s) in LTS
def LTS_sp(L): 
  LTS = max(L)
  ls = []
  for i in range(0, len(L)):
     if L[i] == LTS:
        ls.append(i)
  return ls

"""
Tests that the LTS appears in middle 1/2 by exhaustively testing all binary strings of length n to m 
"""
def testMiddleBin(n, m):
  for i in range(n, m+1):
    print(f"Start {i} ")
    for s in genBinary(i):
      n = len(s)
      L = test_L(s)
      ls = LTS_sp(L)
      ls2 = LTS_sp(L)
      i1 =  math.ceil(n/4 )
      i2 = math.floor((n *3)/4) 
      
      # print(ls)
      for m in ls2: 
        if m < (i1) or m > (i2):
            ls.remove(m)
      if not ls: 
         print(f"uh oh \n {n}, {i1}, {i2} \n {L}\n {ls} \n \n")
    print(f"Finish {i} \n ")
  # print(lst3)

"""
Tests if LTS appears at exactly 1/4 of 3/4 mark
"""
def testEdgeBin(s):
    n = len(s)
    L = test_L(s)
    ls = LTS_sp(L)
    ls2 = LTS_sp(L)
    i1 =  math.ceil(n/4 )
    i2 = math.floor((n *3)/4) 
      
      # print(ls)
    for m in ls2: 
        if m < (i1) or m > (i2):
            ls.remove(m)
    if not ls: 
         print(f"uh oh \n {n}, {i1}, {i2} \n {L}\n {ls} \n \n")


def testNumToH(n, m):
  for i in range(n, m+1):
    print(f"Start {i} ")
    for s in genBinary(i):
      n = len(s)
      L = test_L(s)
      h = 3
      if count_to_h(h, L) == 4*h +3:
         print(s, L)
    print(f"Finish {i} \n ")
  # print(lst3)
   # count = 0
# for s in genBinary(10):
#   count = count + 1
#   if count % 100 == 0: 
#     print(count)
#     # print(f"{s} {test_L(s)}")
#   if test_L(s) == [0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0]:
#      print(s)
# for m in range(1, 5):
#   for n in range(1, 2*m+4):
#     s = test_case3(m, n)
#     plot(s, test_L(s), m, n)
#     plt.show()
#     print(f"(m, n) = ({m}, {n}), \n s = {s} \n \n")
    # print(f"(m, n) = ({m}, {n}), {n} appears {count_n(analyze(s), n)} times")
    # plot(s, test_L(s), 0, 0)
    # plt.show()

# testMiddleBin(3, 20)
# f = open("consecutive ones.txt", "a")
# f.write(f"Testing all binary strings up to length 15")
# f.write(f"Three conseuctive ones:\n {lst3}\n")
# f.write(f"Four conseuctive one:\n {lst4}\n")
# f.write(f"Five conseuctive one:\n {lst5}\n")
# f.write(f"6+ conseuctive one:\n {lst}\n")
# f.close()


def peak_heights(locs, L):
  heights = []
  for p in locs:
      heights.append(L[p])
  return heights
# count_n([0,1,2,1,2,1,2,1,0], 2)

for s in genBinary(1, 10):
  if testEdgeBin(s) == True:
     print(s)
 

# s = test_case4(7)
# L = analyze(s)
# print(L)
# plot(s, L, "")
# plt.show()