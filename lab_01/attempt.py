import numpy as np
import matplotlib.pyplot as plt

#n=5

x_array = [1, 2, 3, 4, 5 ,6]
y_array = [1, 40, 5, 6, 7, 8]
z_array = [[1, 4, 6, 7, 8, 9],
           [5, 5, 6, 7, 8, 9],
           [4, 4, 3, 7, 8, 10],
           [2, 4, 5, 6, 7, 9],
           [1, 5, 6, 7, 8, 9],
           [2, 3, 4, 5, 6, 7]]
#z_array = [1, 4, 6, 7, 8, 9]
p_array = [1, 3, 5, 7, 3, 4]

'''def create_a_mtrx(x_array, p_array, n):
    A = np.ones((n+1, n+1))
    A[0][0] = np.sum(p_array)

    for i in range(1, n + 1):
        # расчет верхнего треугольника
        sum = 0
        for j in range(len(x_array)):
            sum += pow(x_array[j], i) * p_array[j]
        A[0][i] = sum
    h = n
    for i in range(1, h + 1):
        # заполнение верхнего треугольника
        A[i, :h] = A[0, i:]
        h = h - 1

    k = n
    for i in range(1, n+1):
        #расчет нижнего треугольника матрицы
        sum = 0
        for j in range(len(x_array)):
            if n == 1:
                sum += pow(x_array[j], i + 1) * p_array[j]
            else:
                sum += pow(x_array[j], k + 1) * p_array[j]
        k += 1
        A[i][n] = sum
    h = n
    k = n
    for i in range(1, n):
        #заполняем нижний треугольник матрицы(для полиномов 1-3 степеней)
        A[n+1 - len(A[1:h]):, h - 1] = A[1:h, n]
        h -= 1
        k += 1
    return A


def create_b_mtrx(x_array, y_array, p_array, n):
    b = np.ones((n + 1, 1))
    for i in range(n + 1):
        sum = 0
        for j in range(len(y_array)):
            sum += pow(x_array[j], i) * y_array[j] * p_array[j]
        b[i][0] = sum
    return b


A = create_a_mtrx(x_array, p_array, n)
b = create_b_mtrx(x_array, y_array, p_array,n)
print(A)
print(b)

a_vec = np.linalg.inv(A) @ b
if n == 1:
    Y = np.array(x_array) * a_vec[0][0] + a_vec[1][0]
elif n == 2:
    Y = (np.array(x_array) ** 2) * a_vec[0][0] + np.array(x_array) * a_vec[1][0] + a_vec[2][0]
elif n == 3:
    Y = (np.array(x_array) ** 3) * a_vec[0][0] + (np.array(x_array) ** 2) * a_vec[1][0] + np.array(x_array) * a_vec[
        2][0] + a_vec[3][0]
elif n == 4:
    Y = (np.array(x_array) ** 4) * a_vec[0][0] + (np.array(x_array) ** 3) * a_vec[1][0] + (np.array(x_array) ** 2) * \
        a_vec[2][0] + np.array(x_array) * a_vec[3][0] + a_vec[4][0]
elif n == 5:
    Y = (np.array(x_array) ** 5) * a_vec[0][0] + (np.array(x_array) ** 4) * a_vec[1][0] + (np.array(x_array) ** 3) * \
        a_vec[2][0] + (np.array(x_array) ** 2) * a_vec[3][0] + np.array(x_array) * a_vec[4][0] + a_vec[5][0]

print(a_vec)
print(Y)

def scatter_plot(x, y, Y):
    """ отрисовка графика """
    #st.markdown("---")
    fig = plt.figure(figsize=(10,4))
    plt.scatter(x,y)
    plt.plot(x, Y, c='r')
    plt.show()

scatter_plot(x_array, y_array, Y)'''

from matplotlib import pyplot as plt
import numpy as np
#функция
x = np.arange(0.1, 10, 1*(10**-2))
y = np.sin(x) + np.log(x)
#полином 1 степени по функции
p = np.polyfit(x_array,y_array, 2)
#подставляем значения x к полученному полиному
ya = np.polyval(p, x_array)

plt.scatter(x_array, y_array)
plt.plot(x_array, ya)


'''import sys
import string
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


f = open('sdc_point_cloud.txt', 'r')
i=1
a = [0]*19280
x = [0]*19280
y = [0]*19280
z = [0]*19280
for line in f:
    a = line.split("\t")
    if i>2:
        x[i-3] = a[0]
        y[i-3] = a[1]
        z[i-3] = a[2]
    #a[i] = line.split("\n")
    i = i+1
print(x[1])
#print(i)

n=10000
x1 = [0]*n
y1 = [0]*n
z1 = [0]*n
i = 0
while i<n:
    x1[i]=np.float32(x[i]);
    y1[i]=np.float32(y[i]);
    z1[i]=np.float32(z[i]);
    #print(y[i])
    i=i+1

# Plot the surface
#x = np.linspace(-np.pi, np.pi, 50)
#y = x
#z = np.cos(x)

#print(x[1])

fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
ax1 = fig.add_subplot(212, projection='3d')

p = np.polyfit(x_array,y_array, 1)
#подставляем значения x к полученному полиному
ya = np.polyval(p, x_array)


ax.scatter(x_array, y_array, z_array, c='r', marker='o')
ax1.scatter(x_array, ya, z_array, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax1.set_xlabel('X1 Label')
ax1.set_ylabel('Y1 Label')
ax1.set_zlabel('Z1 Label')

plt.show()'''

plt.show()