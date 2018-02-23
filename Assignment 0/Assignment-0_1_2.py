import numpy as np
import matplotlib.pyplot as plt

w=np.random.randn(2,1)
b=np.random.randn()

x_1=[0,0,1,1]
x_2=[0,1,0,1]
X=np.matrix([x_1,x_2])

y=np.transpose(w)*X+np.matrix([[b,b,b,b]])

y_hat=np.heaviside(y,0)

print(y_hat)
#Can not make XOR function, because we need w1+b>1 and w2+b>1 thus w1+w2+2b>2. However we also need w1+w2+b≤0, this means b>2. However if b>2 then from w1+b>1 and w2+b>1 it follows that w1≥-1 and w2≥-1. Thus w1+w2+b>0. Contradiction. Thus we can not make XOR function.

#To get XOR function we need H(b)=0, H(w1+b)=1, H(w2+b)=1 and H(w1+w2+b)=0. This means we need b≤0 and w1+b>0 and w2+b>0 thus w1+w2+2b>0. However we also need w1+w2+b≤0, this means b>0. This is in contradiction with b≤0, so we can not make XOR function this way.
