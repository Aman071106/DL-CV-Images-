a=[1,2,3]
b=a
c=[1,2,3]
print(a==b)
print(c==a)
print(c==b)

c[0]=2

a,b,c

c==a

b[1]=5
a

b==a

import copy

a=[1,2,3]
b=copy.deepcopy(a)
c=copy.copy(a)  #shallow copy

a==b,a==c,b==c

a[0]=9

a==c

b[2]=10

a==b

import copy

a = [(1, 2), (3, 4)]
b = copy.deepcopy(a)
c = copy.copy(a)

# a[0][0] = 99 TypeError: 'tuple' object does not support item assignment
a[0]=(1,5)

print(a)  # [[99, 2], [3, 4]]
print(b)  # [[1, 2], [3, 4]]  <- Deep copy unaffected
print(c)  # [[99, 2], [3, 4]] <- Shallow copy shares inner lists
#  This works as inside elements are mutable, if inside elements are immutable shallowcopy and deepcopy are same

# Explanation:
# Operation	Function	What it Does
# Deep copy	copy.deepcopy(a)	Creates a completely independent copy, including copying all nested mutable elements.
# Shallow copy	copy.copy(a)	Creates a new outer list, but keeps references to the same inner objects (if present).

import copy

a = [[1, 2], [3, 4]]
b = copy.deepcopy(a)
c = copy.copy(a)

a[0][0] = 99

print(a)  # [[99, 2], [3, 4]]
print(b)  # [[1, 2], [3, 4]]  <- Deep copy unaffected
print(c)  # [[99, 2], [3, 4]] <- Shallow copy shares inner lists
#  This works as inside elements are mutable, if inside elements are immutable shallowcopy and deepcopy are same