import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np


def wls_2_dimensional_3(x_array, y_array, p_array, n):
    def create_a_mtrx(x_array, p_array, n):
        A = np.ones((n + 1, n + 1))
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
        for i in range(1, n + 1):
            # расчет нижнего треугольника матрицы
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
            # заполняем нижний треугольник матрицы(для полиномов 1-3 степеней)
            A[n + 1 - len(A[1:h]):, h - 1] = A[1:h, n]
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
    st.write('Матрица А')
    st.write(A)
    st.write('Матрица b')
    b = create_b_mtrx(x_array, y_array, p_array, n)
    st.write(b)


    a_vec = np.linalg.inv(A) @ b
    st.write('Вектор a')
    st.write(a_vec)
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

    return Y

def math(x_array, y_array, p_array, n):

    # полином 1 степени по функции
    p = np.polyfit(x_array, y_array, n, w = np.sqrt(p_array))
    # подставляем значения x к полученному полиному
    ya = np.polyval(p, x_array)

    st.markdown("---")
    fig = plt.figure(figsize=(10, 4))

    plt.scatter(x_array, y_array)
    plt.plot(x_array, ya, c='r')
    st.pyplot(fig)
    st.markdown("---")


def wls_3dimensional_1(x, y, z, p, n):
    # sort data to avoid plotting problems
    x, y, z = zip(*sorted(zip(x, y, z)))

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    data_yz = np.array([y, z])
    data_yz = data_yz.transpose()

    polynomial_features = PolynomialFeatures(degree=n)
    x_poly = polynomial_features.fit_transform(x[:, np.newaxis])

    model = LinearRegression()
    model.fit(x_poly, data_yz, sample_weight=p)
    y_poly_pred = model.predict(x_poly)
    st.write(y_poly_pred)
    return y_poly_pred


def plot_func_3d(x_array, y_array, z_array, Y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = x_array
    y = y_array
    x1, y1 = np.meshgrid(x, y)
    z1 = np.array(Y * z_array)
    #zs = np.array([func(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = z1.reshape(x1.shape)
    ax.plot_surface(x1, y1, Z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    st.write(fig)
    st.pyplot(fig)
    st.markdown("---")


def main():
    st.markdown("### Лабораторная работа 1")
    st.markdown("**Тема**: Построение и программная реализация алгоритма алгоритма наилучшего "
                "среднеквадратичного приближения интерполяции табличных функций")
    st.markdown("**Цель работы** - Получение навыков построения алгоритма реализации метода наименьших квадратов с "
                "использованием полиномов заданной степени в одномерном и двумерном вариантах при аппроксимации "
                "табличных функций с весами")

    st.write("---")

    st.write("Параметры функции:")

    a1, a2, a3, a4 = st.columns(4)
    b1, b2, b3, b4 = st.columns(4)
    c1, c2, c3, c4 = st.columns(4)
    d1, d2, d3, d4 = st.columns(4)
    e1, e2, e3, e4 = st.columns(4)
    f1, f2, f3, f4 = st.columns(4)
    g1, g2, g3, g4 = st.columns(4) # степень полинома

    n = g1.number_input("Введите степень полинома n.", min_value=1, max_value=5, value=1, step=1)

    x1 = a1.number_input("x1", min_value=1.0, max_value=10.0, value=1.0, step=0.25)
    x2 = b1.number_input("x2", min_value=1.0, max_value=10.0, value=1.0, step=0.25)
    x3 = c1.number_input("x3", min_value=1.0, max_value=10.0, value=1.0, step=0.25)
    x4 = d1.number_input("x4", min_value=1.0, max_value=10.0, value=1.0, step=0.25)
    x5 = e1.number_input("x5", min_value=1.0, max_value=10.0, value=1.0, step=0.25)
    x6 = f1.number_input("x6", min_value=1.0, max_value=10.0, value=1.0, step=0.25)

    y1 = a2.number_input("y1", min_value=1.0, max_value=10.0, value=1.0, step=0.25)
    y2 = b2.number_input("y2", min_value=1.0, max_value=10.0, value=1.0, step=0.25)
    y3 = c2.number_input("y3", min_value=1.0, max_value=10.0, value=1.0, step=0.25)
    y4 = d2.number_input("y4", min_value=1.0, max_value=10.0, value=1.0, step=0.25)
    y5 = e2.number_input("y5", min_value=1.0, max_value=10.0, value=1.0, step=0.25)
    y6 = f2.number_input("y6", min_value=1.0, max_value=10.0, value=1.0, step=0.25)

    z1 = a3.number_input("z1", min_value=0.0, max_value=10.0, value=0.0, step=0.25)
    z2 = b3.number_input("z2", min_value=0.0, max_value=10.0, value=0.0, step=0.25)
    z3 = c3.number_input("z3", min_value=0.0, max_value=10.0, value=0.0, step=0.25)
    z4 = d3.number_input("z4", min_value=0.0, max_value=10.0, value=0.0, step=0.25)
    z5 = e3.number_input("z5", min_value=0.0, max_value=10.0, value=0.0, step=0.25)
    z6 = f3.number_input("z6", min_value=0.0, max_value=10.0, value=0.0, step=0.25)

    p1 = a3.number_input("p1", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    p2 = b3.number_input("p2", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    p3 = c3.number_input("p3", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    p4 = d3.number_input("p4", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    p5 = e3.number_input("p5", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    p6 = f3.number_input("p6", min_value=0.0, max_value=10.0, value=0.0, step=0.5)

    x_array = [x1, x2, x3, x4, x5, x6]
    y_array = [y1, y2, y3, y4, y5, y6]
    z_array = [z1, z2, z3, z4, z5, z6]
    p_array = [p1, p2, p3, p4, p5, p6]

    st.subheader("Таблица функции с количеством узлов N:")

    data = {
        "x":x_array,
        "y":y_array,
        "z":z_array,
        "p":p_array,
    }

    result_data = pd.DataFrame(data=data).applymap("{0:.4f}".format)
    st.table(result_data.assign(hack="").set_index("hack"))


    test = st.selectbox("Выберите вид среднеквадратичного приближения", ("Двумерное", "Трёхмерное"))
    if test == "Двумерное":
        Y = wls_2_dimensional_3(x_array, y_array,p_array, n)
        #scatter_plot(x_array, y_array, Y)
        math(x_array, y_array, p_array, n)
    elif test == "Трёхмерное":
        st.write("В процессе доработки...")
        st.write("Матрица Z")
        Y = wls_3dimensional_1(x_array, y_array, z_array, p_array, n)
        st.write(Y)






if __name__ == "__main__":
    main()



