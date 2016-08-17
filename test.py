import theano.tensor as T
from theano import scan
import theano.typed_list as TList
from theano import function
import theano.typed_list as TList
from theano import shared
import numpy as np
import theano

"""
y=np.zeros((10, 1))
z=[]
z.append(y)
print(z)
print("##################")
z.append(y)
print(z)
print("##################")
nmax=np.max(z,axis=0)
print(nmax)
print("##################")
tmax=function([],T.max(z,axis=0))
print(tmax())
print("##################")
"""

#
# y = shared(np.array([[0, 1], [2, 3]],dtype='float32'),name='y')
# y.reshape((2, 2))
#
#
# z=shared(np.array([[0, 1], [2, 3]],dtype='float32'),name='y')
# z.reshape((2, 2))
#
# temp=TList.make_list([y,z])
#
# print(temp.eval())

# def z_calc(i):
#     print(i)
#     return T.fscalar()
#
# arr=[1,2,3]
#
# s,_= scan(z_calc, sequences=[T.arange(len(arr), dtype='int32')])
#
# f=function([],s)
# f()

y = shared(np.array([[0, 1], [2, 3]], dtype='float32'), name='y')
y.reshape((2, 2))

t = TList.TypedListType(T.fmatrix)()
t.append(y)

print(t.eval())