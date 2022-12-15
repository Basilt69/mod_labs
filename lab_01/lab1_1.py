import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np






def main():
    st.markdown("### Лабораторная работа 1 - Построение и программная реализация алгоритма алгоритма наилучшего "
                "среднеквадратичного приближения "
                "интерполяции табличных функций")
    st.markdown("Цель работы - Получение навыков построения алгоритма реализации метода наименьших квадратов с "
                "использованием полиномов заданной степени в одномерном и двумерном вариантах при аппроксимации "
                "табличных функций с весами")

    st.write("---")

    st.write("Параметры функции:")

    a1, a2, a3 = st.columns(3)
    b1, b2, b3 = st.columns(3)
    c1, c2, c3 = st.columns(3)
    d1, d2, d3 = st.columns(3)
    e1, e2, e3 = st.columns(3)
    f1, f2, f3 = st.columns(3)

    x1 = a1.number_input("x1", min_value=1, max_value=10, value=1, step=1)
    x2 = b1.number_input("x2", min_value=1, max_value=10, value=1, step=1)
    x3 = c1.number_input("x3", min_value=1, max_value=10, value=1, step=1)
    x4 = d1.number_input("x4", min_value=1, max_value=10, value=1, step=1)
    x5 = e1.number_input("x5", min_value=1, max_value=10, value=1, step=1)
    x6 = f1.number_input("x6", min_value=1, max_value=10, value=1, step=1)

    y1 = a2.number_input("y1", min_value=1, max_value=10, value=1, step=1)
    y2 = b2.number_input("y2", min_value=1, max_value=10, value=1, step=1)
    y3 = c2.number_input("y3", min_value=1, max_value=10, value=1, step=1)
    y4 = d2.number_input("y4", min_value=1, max_value=10, value=1, step=1)
    y5 = e2.number_input("y5", min_value=1, max_value=10, value=1, step=1)
    y6 = f2.number_input("y6", min_value=1, max_value=10, value=1, step=1)

    p1 = a3.number_input("p1", min_value=1, max_value=10, value=1, step=.5)
    p2 = b3.number_input("p2", min_value=1, max_value=10, value=1, step=1)
    p3 = c3.number_input("p3", min_value=1, max_value=10, value=1, step=1)
    p4 = d3.number_input("p4", min_value=1, max_value=10, value=1, step=1)
    p5 = e3.number_input("p5", min_value=1, max_value=10, value=1, step=1)
    p6 = f3.number_input("p6", min_value=1, max_value=10, value=1, step=1)

    x_array = [x1, x2, x3, x4, x5, x6]
    y_array = [y1, y2, y3, y4, y5, y6]
    p_array = [p1, p2, p3, p4, p5, p6]
    st.subheader("Таблица функции с количеством узлов N:")

    data = {
        "x":x_array,
        "y":y_array,
        "p":p_array,
    }

    result_data = pd.DataFrame(data=data).applymap("{0:.4f}".format)
    st.table(result_data.assign(hack="").set_index("hack"))



if __name__ == "__main__":
    main()



