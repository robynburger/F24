import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import math

# max size of the test string 
max_length = 20

# number of test cases
num_tests = 100000

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
  A = populate_A(D, n)
  p = find_p(F, s)
  return find_L(F, s)[0]

def test_rand(alphabet, max_length):
  for x in range(num_tests):
      if x % 1000 == 0: print(f"Test {x}")
      s = ""
      for _ in range(random.randint(max_length-5, max_length)):
        s += str(random.choice(alphabet))
 
 
 
def genBinary(n):
    return [''.join(p) for p in product('01', repeat=n)]


def LCS(s1, s2):
  m = len(s1)
  n = len(s2)
  dp = [[0] * (n + 1) for x in range(m + 1)]
  for i in range(1, m + 1):
    for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j],
                               dp[i][j - 1])
  return dp[m][n]

"""
approximates LTS
"""
def approxRandLTS(s, i1, i2):
      # print(f"s= {s} \n i1 = {i1} i2 = {i2}")
      # print(f"s = {s} \n")
      i = random.randint(i1 +1, i2 -1)
      approx = float(LCS(s[:i], s[i:]))
      actual = float(max(LTS(s)))
      return approx/actual

'''
Finds the average (over all binary strings) 
approximation for a random sp in middle 1/2 
'''
def testRandAllBinary(n, m):
  count = float(1)
  total = 1
  for i in range(n, m+1):
    print(f"Start {i}")
    for s in genBinary(i):
      count += 1
      n = len(s)
      i1 =  math.ceil(n/4 ) 
      i2 = math.floor((n *3)/4) 
      total += approxRandLTS(s, i1, i2)
  return total/count



"""
approximates LTS
"""
def approxAvgLTS(s, i1, i2):
      # print(f"s= {s} \n i1 = {i1} i2 = {i2}")
      # print(f"s = {s} \n")
      i = random.randint(i1 +1, i2 -1)
      approx = float(LCS(s[:i], s[i:]))
      actual = float(max(LTS(s)))
      return approx/actual

'''
Finds the average (over all binary strings) 
approximation for a random sp in middle 1/2 
'''
def testAvgAllBinary(n, m):
  count = 0.0
  total = 0.0
  # for i in range(n, m+1):
  #   print(f"Start {i}")
  for s in ["00000000000", "000000111111"]:
      count += 1.0
      actual = float(max(LTS(s)))
      n = len(s)
      i1 =  math.ceil(n/4 ) 
      i2 = math.floor((n *3)/4) 
      inner_total = 0.0
      inner_count = 0.0
      for j in range(i1, i2+1):
        inner_count += 1.0
        inner_total += float(LCS(s[:j], s[j:]))
      avgApprox = inner_total/inner_count
      actual = float(max(LTS(s)))
      ratio = avgApprox/actual
        # if max(LTS(s)) > 6:
      print(s, LTS(s), testEdgeBin(s))
      print("l1 =", LTS(s)[i1], " and l2 =", LTS(s)[i2], "at i1 =", i1,"i2 =", i2 )
      print("innertotal", inner_total)
      print("approx", avgApprox, "/", "actual", actual, "=", ratio, "\n")
      total += ratio


# returns splitpoint(s) that result(s) in LTS
def LTS_sp(L): 
  LTS = max(L)
  ls = []
  for i in range(0, len(L)):
     if L[i] == LTS:
        ls.append(i)
  return ls

"""
Tests if LTS appears at exactly 1/4 of 3/4 mark
"""
def testEdgeBin(s):
    n = len(s)
    L = LTS(s)
    ls = LTS_sp(L)
    ls2 = LTS_sp(L)
    i1 =  math.ceil(n/4 )
    i2 = math.floor((n *3)/4) 
      
      # print(ls)
    for m in ls2: 
        if m < (i1) or m > (i2):
            ls.remove(m)
    if not ls: 
         return False
    return True
      
      # total += approxRandLTS(s, i1, i2)
  # return total/count



testAvgAllBinary(1, 2) 

