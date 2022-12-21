import numpy as np
import matplotlib.pyplot as plt

n=1

x_array = [1, 2, 3, 4, 5 ,6]
y_array = [1, 4, 5, 6, 7, 8]
p_array = [1, 3, 5, 7, 3, 4]

def create_a_mtrx(x_array, p_array, n):
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

scatter_plot(x_array, y_array, Y)