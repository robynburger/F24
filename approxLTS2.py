"""
Idea: take max of L_(n/4) L_(3n/4) and random from middle half
"""

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
            if s[i-1] == s[k-1]: 
              F[k][i][j] = F[k-1][i-1][j] + 1 
            else:
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
    if curr > maxVal: 
      maxVal = curr 
      p = i
  return p

"""
Finds F_n(i, i+1) for each i 

"""
def find_L(F, s): 
  n = len(s)
  L = np.zeros((1, n+1), dtype = int)
  n = len(s) 
  for i in range(1, n):       
    L[0][i] = F[n-1][i-1][i] 
  return L
    
'''
Prints F, D, and A for input string 's'
'''
def LTS(s):
  n = len(s)
  empty_bufferF = np.zeros((n+1, n+1, n+1), dtype=int)
  bufferF = populate_F(empty_bufferF, s)
  D = populate_D(bufferF, n)
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



'''
Finds the average (over all binary strings) 
approximation for a random sp in middle 1/2 
'''
def testAvgAllBinary(n, m):
  count = 0.0
  total = 0.0
  min = 0
  for i in range(n, m+1):
    print(f"Start {i}", flush=True)
    for s in genBinary(i):
      count += 1.0
      actual = float(max(LTS(s)))
      n = len(s)
      i1 =  math.ceil(n/4 ) 
      i2 = math.floor((n *3)/4)
      l1 = float(LCS(s[:i1], s[i1:]))
      l2 = float(LCS(s[:i2], s[i2:]))
  
      j1 = int(i1  + max(1, l2-l1 + 1, i1-l1))
      j2 = int(i2 - max(1, l1-l2 + 1, i1-l2))
      # j1 = int(i1 + 1 + max(0 , l2-l1))
      # j2 = int(i2 -  1 - max(0, l1-l2))
      inner_total = 0.0
      inner_count = 0.0
      if j1 > j2 :
        rand_mid = 0
      else:
        for j in range(j1, j2+1):
          inner_count += 1.0
          inner_total += float(LCS(s[:j], s[j:]))         
        rand_mid = inner_total/inner_count
      approx = max(l1, l2, rand_mid)
      actual =  float(max(LTS(s)))
      ratio = approx/actual
      if ratio < 2/3:
          print(s, LTS(s), flush=True)
          print("i1", i1, "i2", i2)
          print("j1", j1, "j2", j2)
          print("approx", approx, "/", "actual", actual, "=", ratio, flush=True)
          print("l1", l1, "l2", l2, f"rand from {i1} to {i2}", rand_mid, flush=True)
          print("\n", flush=True)
      total += ratio
    print(f"End {i}\n", flush=True)
  return total/count


print(testAvgAllBinary(16, 30))   

