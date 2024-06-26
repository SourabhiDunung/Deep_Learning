

1. Importing Libraries
"""

#importing libraries
import numpy as np
import tensorflow as ts

"""2. Defining Step Function"""

#
def stepfun(v):
  if v>=0:
    return 1
  else:
    return 0

"""3. defining perceptron using matmul"""

#
def percepfun(x,w,b):
  v=np.matmul(w,x)+b
  y=stepfun(v)
  return y

"""LINEAR FUNCTIONS

with numpy library

AND LOGIC
"""

def ANDlogic(x):
  w=np.array([1,1])
  b=-1.5
  return percepfun(x,w,b)

test1=np.array([0.0,0.0])
test2=np.array([0.0,1.0])
test3=np.array([1.0,0.0])
test4=np.array([1.0,1.0])

y1=ANDlogic(test1)
y2=ANDlogic(test2)
y3=ANDlogic(test3)
y4=ANDlogic(test4)

print("AND (0,0) = ",y1)
print("AND (0,1) = ",y2)
print("AND (1,0) = ",y3)
print("AND (1,1) = ",y4)

"""OR LOGIC"""

def ORlogic(x):
  w=np.array([1,1])
  b=-0.5
  return percepfun(x,w,b)

test1=np.array([0.0,0.0])
test2=np.array([0.0,1.0])
test3=np.array([1.0,0.0])
test4=np.array([1.0,1.0])

y1=ORlogic(test1)
y2=ORlogic(test2)
y3=ORlogic(test3)
y4=ORlogic(test4)

print("OR (0,0) = ",y1)
print("OR (0,1) = ",y2)
print("OR (1,0) = ",y3)
print("OR (1,1) = ",y4)

"""NAND LOGIC"""

def NANDlogic(x):
  w=np.array([-1,-1])
  b=1.5
  return percepfun(x,w,b)

test1=np.array([0.0,0.0])
test2=np.array([0.0,1.0])
test3=np.array([1.0,0.0])
test4=np.array([1.0,1.0])

y1=NANDlogic(test1)
y2=NANDlogic(test2)
y3=NANDlogic(test3)
y4=NANDlogic(test4)

print("NAND (0,0) = ",y1)
print("NAND (0,1) = ",y2)
print("NAND (1,0) = ",y3)
print("NAND (1,1) = ",y4)

"""NON-LINEAR FUNCTION

XOR LOGIC
"""

def XORlogic(x):
  #w=np.array([1,1])
  #b=1.5
  h1=ORlogic(x)
  h2=NANDlogic(x)
  final_x=np.array([h1,h2])
  final_output=ANDlogic(final_x)
  return final_output

test1=np.array([0.0,0.0])
test2=np.array([0.0,1.0])
test3=np.array([1.0,0.0])
test4=np.array([1.0,1.0])

y1=XORlogic(test1)
y2=XORlogic(test2)
y3=XORlogic(test3)
y4=XORlogic(test4)

print("XOR (0,0) = ",y1)
print("XOR (0,1) = ",y2)
print("XOR (1,0) = ",y3)
print("XOR (1,1) = ",y4)

"""WRIRTING A USER DEFINED CODE"""

w=float(input("Enter 1st bit: "))
x=float(input("ENter 2nd bit: "))
test=np.array([w,x])

t1=ANDlogic(test)
t2=ORlogic(test)
t3=NANDlogic(test)
t4=XORlogic(test)

print("AND: ",t1)
print("OR: ",t2)
print("NAND: ",t3)
print("XOR: ",t4)
