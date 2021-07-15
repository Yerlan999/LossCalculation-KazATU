# Задание значении характеристик материала для фаз и троса
Faza_A = Fazy("Фаза А", 35.336, 1.0, 240.0, 0.0, 19.0)
Faza_B = Fazy("Фаза В", 35.336, 1.0, 240.0, 6.3, 19.0)
Faza_C = Fazy("Фаза С", 35.336, 1.0, 240.0, 4.2, 25.0)
Tross = Fazy("Тросс", 17.336, 8.0, 240.0, 2.1, 28.0)


# В данном случае у нас будет 50 гармоник от [1 до 50]
garmoniki = Garmoniki(50)


# Создание объекта для проведения расчетов
rashet = Rachety()


# Вычисление погонных активных сопротивлении для фаз и троса.
# Получаем матрицу с размерностью [4 на 50]
pogon_aktiv_soprotiv_faz = rashet.pogon_aktiv_soprotiv(Fazy, garmoniki)


# Вычисление погонных индуктивных сопротивлении для фаз и троса.
# Получаем матрицу с размерностью [4 х 4 х 50]
inductiv_soprotiv_faz = rashet.inductiv_soprotiv(Fazy, garmoniki)


# Вычисление погонных ёмкостных проводимостей для фаз и троса.
# Получаем матрицу с размерностью [4 х 4 х 50]
emkostn_provodimost_faz = rashet.emkostnaya_provodimost(Fazy, garmoniki)


# Матрица полных погонных прводимостей
Y = rashet.polnaya_provodimost(emkostn_provodimost_faz)


# Матрица полных погонных сопротивлении
Z = rashet.polnoe_soprotivlenye(inductiv_soprotiv_faz, pogon_aktiv_soprotiv_faz)


# Функция для расчета комплексных квадратных матриц лямбда_напряж и лямбда_ток
lambda_U, lambda_I = rashet.calculate_lambdas(Z, Y)


# Функция для расчета комплексных квадратных матриц экспонен_напряжения и экспонента_ток
# Для примера были взяты значения для отрезка 300
pos_expon_U, neg_expon_U = rashet.calculate_exponentials(lambda_U, 300)
pos_expon_I, neg_expon_I = rashet.calculate_exponentials(lambda_I, 300)
