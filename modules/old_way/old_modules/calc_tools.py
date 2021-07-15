from numpy import linalg as LA
from scipy.linalg import sqrtm
import pandas as pd
from itertools import tee
import numpy as np
import scipy
import math


# Класс для инициализации параметров всей системы
class System():

    """
    Для инициализации системы вводятся следующие параметры в порядке упоминания
        (в скобках указаны тип данных):

        1) Наименование подстанции (строка)
        2) Наименование присоединения (строка)
        3) Количество присоединении (целое число)
        4) Длина линии (дробное или целое число) в километрах
        5) Количество проводов (целое число) вкючая трос
        6) Количество измерении (целое число)
        7) Интервал измерении (дробное или целое число) в минутах
        8) Устройство опоры ЛЭП Одноцепная/Двухцепная (булевое значение)
        9) Метод выполнения фазы. Разщеплена/Неразщеплена (булевое значение)
        10) Форма разщепленной фазы. Правильный треугольник/Нет (булевое значение)
    """

    def __init__(self, naimen_podstan, naimen_prisoedin, kol_prisoedin, dlina_linii,
     kol_provodov, kol_izmeren, interval, dvuh_tsepnaya=False, rassheplena=False, prav_treugolnik=False):

        self.naimen_podstan = naimen_podstan
        self.naimen_prisoedin = naimen_prisoedin
        self.kol_prisoedin = kol_prisoedin
        self. dlina_linii =  dlina_linii
        self.kol_provodov = kol_provodov
        self.kol_izmeren = kol_izmeren
        self.interval = interval
        self.dvuh_tsepnaya = dvuh_tsepnaya
        self.rassheplena = razsheplena
        self.prav_treugolnik = prav_treugolnik



# Класс для инициализации параметров проводов системы (фаз и троса)
class Fazy():

    # Список для хранения объектов(наименовании) фаз и тросса
    spisok_provodov = []

    """
    Для инициализации фаз и тросов вводятся следующие параметры в порядке упоминания
        (в скобках указаны тип данных):

        1) Наименование фазы (строка)
        2) Значение гаммы, или удельной проводимости (дробное или целое число)
        3) Значение магнитной проницаемости (дробное или целое число)
        4) Значение поперечного сечения провода (дробное или целое число) в мм2
        5) Координата Х для провода (дробное или целое число) в мертах
        6) Координата У для провода (дробное или целое число) в мертах
                          !!! Применчание !!!
                Один из проводов выбирается как основное
    """

    def __init__(self, name, gamma, magnit_pronitsaemost, poper_sechenie, x, y):
        self.name = name
        self.gamma = gamma
        self.magnit_pronitsaemost = magnit_pronitsaemost
        self.poper_sechenie = poper_sechenie
        self.radius = (math.sqrt(self.poper_sechenie/math.pi))/1000
        self.x = x
        self.y = y

        # Автоматическое добавление инициируемого провода в список
        self.dobavit_provod(self)

    # Классовая функция для добавления созданного провода в список
    @classmethod
    def dobavit_provod(cls, faza):
        cls.spisok_provodov.append(faza)



# Класс для инициализации параметров гармоник
class Garmoniki():

    """
    Для инициализации фаз и тросов вводятся следующие параметры в порядке упоминания
        (в скобках указаны тип данных):

        1) Количество гармоник (целое число) включая основное и выше
    """

    def __init__(self, chislo_garmonik, ):
        self.chislo_garmonik = chislo_garmonik
        # w = 2 * pi * f для каждой гармоники [от 1-ой до указанной]!!!
        # создение списка с частотами всех созданных гармоник
        self.chastoty_garmonic = [2 * math.pi * (50 * w) for w in range(1, chislo_garmonik+1)]

    # Позволяет доставать значение частоты гармоники по номеру. Пример: Объект[номер гармоники]
    def __getitem__(self, garmonika):
        if garmonika > 0 and garmonika <= self.chislo_garmonik:
            return self.chastoty_garmonic[garmonika-1]

    # Позволяет проитерировать через гармоники(их частоты)
    def __iter__(self):
        self.index = 0
        return self

    # Позволяет проитерировать через гармоники(их частоты)
    def __next__(self):
        if self.index < self.chislo_garmonik:
            garmonica = self.chastoty_garmonic[self.index]
            self.index += 1
        else:
            raise StopIteration
        return garmonica

    # Позволяет получать количество созданных гармоники
    def __len__(self):
        return self.chislo_garmonik


# Класс для инициализации методов для произведения расчетов в системе
class Rachety():

    # Вычисляет матрицу активных погонных сопротивлении.
    # Возвращает матрицу с размерностью 4х50. Каждая строка - фаза. Каждый столбец - значение
    # активного сопротивления для каждой гармоники
    def pogon_aktiv_soprotiv(self, fazy, garmoniki):

        #Matritsa_Pog_Soprot = list()
        matritsa = list()
        for faza in fazy.spisok_provodov:
            #print(faza.name)
            Ri = (math.sqrt(faza.poper_sechenie/math.pi))/1000
            Roi = 1000/(faza.gamma*faza.poper_sechenie)
            znach_pog_soprot = list()

            for garmonika in garmoniki:
                Xwi = (Ri/2) * math.sqrt((garmonika * faza.gamma * faza.magnit_pronitsaemost)/2)
                if Xwi < 1:
                    Rpi = Roi * (1 + math.pow(Xwi, 4/3))
                    znach_pog_soprot.append(Rpi)
                elif Xwi >= 1 and Xwi < 30:
                    Rpi = Roi * (Xwi + 0.25 + (3/64 * Xwi))
                    znach_pog_soprot.append(Rpi)
                else:
                    Rpi = Roi * (Xwi + 0.265)
                    znach_pog_soprot.append(Rpi)

            matritsa.append(znach_pog_soprot)
        return np.array(matritsa)


    # Вычисление матрицы для индуктивных взаимных и собственных сопротивлении.
    # Возвращает матрицу с размерностью 4х4х50. Строки и столбцы - фазы. т.е (пересечение)
    # взаимные или собственные индуктивности. Слои - для каждой гармоники.
    def inductiv_soprotiv(self, fazy, garmoniki):
        general_list = list()
        for faza_i in fazy.spisok_provodov:
            temporal_list = list()
            for faza_j in fazy.spisok_provodov:
                if faza_i.name == faza_j.name:
                    # !!! ВОПРОС !!! РАДИУС ПРОВОДА ИЛИ ДИАМЕТР ПРОВОДА ???????
                    R = 1000/faza_i.radius                     # Радиус провода в метрах!!
                    temporal_list.append(R)
                else:
                    t2 = (math.sqrt(((faza_i.x - faza_j.x)**2 + (faza_i.y - faza_j.y)**2)))
                    D = 1000/t2
                    temporal_list.append(D)
            general_list.append(temporal_list)
        two_dim_array = np.array(general_list)

        # Умножение 2-х мерной матрицы на каждую гармонику для создания 3-х мерной матрицы
        temp_array = list()
        for gar in garmoniki:
            res = two_dim_array * gar
            temp_array.append(res)
        three_dim_array = np.array(temp_array)

        # Применение логарифмической функции для каждой ячейки 3-х мерной матрицы

        return np.dot(np.log10(three_dim_array), 0.145)

    # Вычисление матрицы для ёмкостных взаимных и собственных проводимостей.
    # Возвращает матрицу с размерностью 4х4х50. Строки и столбцы - фазы. т.е (пересечение)
    # взаимные или собственные индуктивности. Слои - для каждой гармоники.
    def emkostnaya_provodimost(self, fazy, garmoniki):
        general_list = list()
        for faza_i in fazy.spisok_provodov:
            temporal_list = list()
            for faza_j in fazy.spisok_provodov:
                if faza_i.name == faza_j.name:
                    R = 2*faza_i.y/faza_i.radius
                    temporal_list.append(R)
                else:
                    t2 = (math.sqrt(((faza_i.x - faza_j.x)**2 + (faza_i.y - faza_j.y)**2)))
                    D = t2/t2
                    temporal_list.append(D)
            general_list.append(temporal_list)
        two_dim_array = np.array(general_list)

        # Применение логарифмической функции для каждой ячейки 3-х мерной матрицы
        output_matrix = np.dot(np.log10(two_dim_array), 41.4*10**6)

        # Инвертирования матрицы
        inverted_2D_array = np.linalg.inv(output_matrix)

        # Умножение 2-х мерной матрицы на каждую гармонику для создания 3-х мерной матрицы
        temp_array = list()
        for gar in garmoniki:
            res = inverted_2D_array * gar
            temp_array.append(res)
        three_dim_array = np.array(temp_array)
        return three_dim_array

    # # Позволяет расчитывать полное погонное сопротивление. На выходе матрица [4x4x50]
    def polnoe_soprotivlenye(self, matritsa_induc_soprotiv, pogon_aktiv_soprotiv_faz):
        imagiary = lambda cell: np.complex(0, cell)
        vectorized_3D_array = np.vectorize(imagiary)
        output_matrix = vectorized_3D_array(matritsa_induc_soprotiv)

        zeros = np.zeros(matritsa_induc_soprotiv.shape, dtype=matritsa_induc_soprotiv.dtype)
        for i in range(pogon_aktiv_soprotiv_faz.shape[0]):
            zeros[:, i, i] = pogon_aktiv_soprotiv_faz[i]
        result_matrix = output_matrix + zeros
        return result_matrix


    # Позволяет расчитывать полную погонную проводимость. На выходе матрица [4x4x50]
    def polnaya_provodimost(self, matritsa_emcost_provodim):
        imagiary = lambda cell: np.complex(0, cell)
        vectorized_3D_array = np.vectorize(imagiary)
        output_matrix = vectorized_3D_array(matritsa_emcost_provodim)
        return output_matrix



# Методы для специальной парной итерации
def pairwise_1(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def pairwise_2(iterable):
    a = iter(iterable)
    return zip(a, a)


class Looper():
    def __init__(self, start, end, step):
        self.start = start
        self.end = end
        self.step = step
        self.counter = 0
        self.result = 0

    def __iter__(self):
        return self
    def __next__(self):
        if self.result < self.end:
            self.result = self.start + self.step*self.counter
            self.counter += 1
            return round(self.result, 3)
        else:
            raise StopIteration
