import streamlit as st

from lab_01 import lab1_1
from lab_02 import lab2_1
from nirs import nir


# st.set_page_config(initial_sidebar_state="collapsed")
st.sidebar.image('logo.png', width=300)


def header():
    author = '''
    made by [Basil Tkachenko](https://github.com/Basilt69)
    in [BMSTU](https://bmstu.ru)
    '''

    st.header('МГТУ им. Баумана, Кафедра ИУ7')
    st.markdown("**Курс:** Моделирование")
    st.markdown("**Преподаватель:** Градов В.М.")
    st.markdown("**Студент:** Ткаченко В.М.")
    st.sidebar.markdown(author)


def main():
    header()
    lab = st.sidebar.radio(
        "Выберите лабораторную работу:", (
            "1. Алгоритм наилучшего среднеквадратичного приближения.",
            "2. Численное интегрирование.",
            "3 НИР(Метод Монте-Карло).",
        ),
        index=2
    )

    if lab[:1] == "1":
        lab1_1.main()

    elif lab[:1] == "2":
        lab2_1.main()

    elif lab[:1] == "3":
        nir.main()


if __name__ == "__main__":
    main()