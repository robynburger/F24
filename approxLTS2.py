"""
Idea: take max of L_(n/4) L_(3n/4) and random from middle half
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import math
 
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

def LTS(s):
  n = len(s)
  L = []
  for l in range(0, n+1):
    L.append(LCS(s[:l], s[l:]))
  return np.array(L)

def testAvgAllBinary(n, m):
  count = 0.0
  total = 0.0
  min = 0
  for p in range(n, m+1):
    i = 2*p
    # print(f"Start {i}", flush=True)
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
      if ratio < 0.5:
          print(s, LTS(s), flush=True)
          print("i1", i1, "i2", i2, flush=True)
          print("j1", j1, "j2", j2, flush = True)
          print("approx", approx, "/", "actual", actual, "=", ratio, flush=True)
          print("l1", l1, "l2", l2, f"rand from {i1} to {i2}", rand_mid, flush=True)
          print("\n", flush=True)
          return 0

def testRandomBinary(n, m):
  for count in range(1000000000000000000000000):
      if count % 100000 == 0:
         print(count, flush=True)
      s = []
      for length in range(random.randint(n,m)):
        s.append(random.choice(["0","1"]))
      s = "".join(s)

      actual = float(max(LTS(s)))
      n = len(s)
      i1 =  math.ceil(n/4 ) 
      i2 = math.floor((n *3)/4)
      l1 = float(LCS(s[:i1], s[i1:]))
      l2 = float(LCS(s[:i2], s[i2:]))
  
      j1 = int(i1  + max(1, l2-l1 + 1, i1-l1))
      j2 = int(i2 - max(1, l1-l2 + 1, i1-l2))

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
          print("i1", i1, "i2", i2, flush=True)
          print("j1", j1, "j2", j2, flush = True)
          print("approx", approx, "/", "actual", actual, "=", ratio, flush=True)
          print("l1", l1, "l2", l2, f"rand from {i1} to {i2}", rand_mid, flush=True)
          print("\n", flush=True)




print(testAvgAllBinary(5, 20))   

