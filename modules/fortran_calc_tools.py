from numpy import linalg as LA
from scipy.linalg import logm, expm, sqrtm, eig, lu, lu_factor, lu_solve
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

    def __init__(self, podstans_object, harmonics_object, reshet_object, fazy_class, naimen_podstan, naimen_prisoedin, kol_prisoedin, dlina_linii,
     kol_provodov, kol_izmeren, interval, dvuh_tsepnaya=False, rassheplena=False, prav_treugolnik=False):

        self.podstans_object = podstans_object
        self.harmonics_object = harmonics_object
        self.reshet_object = reshet_object
        self.fazy_class = fazy_class
        self.naimen_podstan = naimen_podstan
        self.naimen_prisoedin = naimen_prisoedin
        self.kol_prisoedin = kol_prisoedin
        self.dlina_linii =  dlina_linii
        self.kol_provodov = kol_provodov
        self.kol_izmeren = kol_izmeren
        self.interval = interval
        self.dvuh_tsepnaya = dvuh_tsepnaya
        self.rassheplena = rassheplena
        self.prav_treugolnik = prav_treugolnik

    def line_matrix(self, pris_num=1):
        assert type(pris_num) == int, "Must be integer"

        UM, FUM, AIM, FIM = [], [], [], []
        AU, PU, AI, PI = [], [], [], []
        THD_U, THD_I, RMSU, RMSI = [], [], [], []

        for att_name, obj in getattr(self.podstans_object, "prisoed_"+str(pris_num)).__dict__.items():
            if att_name.startswith("faza_") and type(obj.df) != type(None):
                faza_x = getattr(getattr(self.podstans_object, "prisoed_"+str(pris_num)), att_name)
                um, fum, aim, fim = faza_x.get_harmonic()
                au, pu, ai, pi = faza_x.get_main_harm()
                thd_u, thd_i, rmsu, rmsi = faza_x.get_rms()
                UM.append(um); FUM.append(fum); AIM.append(aim); FIM.append(fim);
                AU.append(au); PU.append(pu); AI.append(ai); PI.append(pi);
                THD_U.append(thd_u); THD_I.append(thd_i); RMSU.append(rmsu); RMSI.append(rmsi);

        # Returns 3 dimensional matrix [timeindex, harmonics, phase] or 2 dim-al [timeindex, phase]
        return np.swapaxes(np.array(UM),0,2), np.swapaxes(np.array(FUM),0,2), np.swapaxes(np.array(AIM),0,2), np.swapaxes(np.array(FIM),0,2), np.swapaxes(np.array(AU),0,1), np.swapaxes(np.array(PU),0,1), np.swapaxes(np.array(AI),0,1), np.swapaxes(np.array(PI),0,1), np.swapaxes(np.array(THD_U),0,1), np.swapaxes(np.array(THD_I),0,1), np.swapaxes(np.array(RMSU),0,1), np.swapaxes(np.array(RMSI),0,1)

    def line_matrixWithPrisoed(self):

        pUM, pFUM, pAIM, pFIM = [], [], [], []
        pAU, pPU, pAI, pPI = [], [], [], []
        pTHD_U, pTHD_I, pRMSU, pRMSI = [], [], [], []

        for pris_name, pris in self.podstans_object.__dict__.items():
            if pris_name.startswith("prisoed_"):
                iUM, iFUM, iAIM, iFIM = [], [], [], []
                iAU, iPU, iAI, iPI = [], [], [], []
                iTHD_U, iTHD_I, iRMSU, iRMSI = [], [], [], []

                for att_name, obj in pris.__dict__.items():
                    if att_name.startswith("faza_") and type(obj.df) != type(None):
                        faza_x = getattr(pris, att_name)
                        um, fum, aim, fim = faza_x.get_harmonic()
                        au, pu, ai, pi = faza_x.get_main_harm()
                        thd_u, thd_i, rmsu, rmsi = faza_x.get_rms()

                        iUM.append(um); iFUM.append(fum); iAIM.append(aim); iFIM.append(fim);
                        iAU.append(au); iPU.append(pu); iAI.append(ai); iPI.append(pi);
                        iTHD_U.append(thd_u); iTHD_I.append(thd_i); iRMSU.append(rmsu); iRMSI.append(rmsi);

                pUM.append(np.array(iUM)); pFUM.append(np.array(iFUM)); pAIM.append(np.array(iAIM)); pFIM.append(np.array(iFIM));
                pAU.append(np.array(iAU)); pPU.append(np.array(iPU)); pAI.append(np.array(iAI)); pPI.append(np.array(iPI));
                pTHD_U.append(np.array(iTHD_U)); pTHD_I.append(np.array(iTHD_I)); pRMSU.append(np.array(iRMSU)); pRMSI.append(np.array(iRMSI));

        return pUM, pFUM, pAIM, pFIM, pAU, pPU, pAI, pPI, pTHD_U, pTHD_I, pRMSU, pRMSI



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

    @classmethod
    def construc_matrices(cls):
        XA, YA, OMP, GM, S = [], [], [], [], []

        for faza in cls.spisok_provodov:
            XA.append(faza.x)
            YA.append(faza.y)
            GM.append(faza.gamma)
            OMP.append(faza.magnit_pronitsaemost)
            S.append(faza.poper_sechenie)

        return np.array(XA, dtype=type), np.array(YA, dtype=type), np.array(OMP, dtype=type), np.array(GM, dtype=type), np.array(S, dtype=type)



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

        matritsa = list()
        for faza in fazy.spisok_provodov:
            Ri = (math.sqrt(faza.poper_sechenie/math.pi))/1000
            Roi = 1000/(faza.gamma*faza.poper_sechenie)
            znach_pog_soprot = list()

            for garmonika in garmoniki:
                Xwi = (Ri/2) * math.sqrt((garmonika * faza.gamma * faza.magnit_pronitsaemost)/2)
                if Xwi <= 1:
                    Rpi = Roi * (1 + math.pow(Xwi, 4/3))
                    znach_pog_soprot.append(Rpi)
                elif Xwi > 1 and Xwi < 30:
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
                    R = 1000/faza_i.radius                     # Радиус провода в метрах!!
                    temporal_list.append(R)
                else:
                    t2 = (math.sqrt(((faza_i.x - faza_j.x)**2 + (faza_i.y - faza_j.y)**2)))
                    D = 1000/t2
                    temporal_list.append(D)
            general_list.append(temporal_list)
        two_dim_array = np.array(general_list)

        two_dim_array = np.dot(np.log10(two_dim_array), 0.145)

        # Умножение 2-х мерной матрицы на каждую гармонику для создания 3-х мерной матрицы
        temp_array = list()
        for gar in garmoniki:
            res = two_dim_array * gar
            temp_array.append(res)
        three_dim_array = np.array(temp_array)

        # Применение логарифмической функции для каждой ячейки 3-х мерной матрицы

        return three_dim_array

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

        # Применение логарифмической функции для каждой ячейки 2-х мерной матрицы
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


    # Функция для расчета комплексных квадратных матриц лямбда_напряж и лямбда_ток
    def calculate_lambdas(self, Z, Y): # <----!ВАЖНО! "Сначала Z, потом Y"

        # Проверка правильности входных матриц
        assert len(Z.shape) == len(Y.shape) and len(Z.shape) == 3, "Both Z and Y must be 3 dimentional matrices"
        for dim in range(len(Z.shape)):
            assert Z.shape[dim] == Y.shape[dim], "Z and Y must be equal in size"
        assert Z.shape[1] == Z.shape[2], "Z and Y must be sqare matrices"

        lambda_U_main_matrix = []
        lambda_I_main_matrix = []

        ###   Расчет матрицы лямбда напряжения !!!
        for harm in range(Z.shape[0]):
            list_of_matrices = []
            list_of_deters = []
            Au = np.dot(Z[harm], Y[harm])
            eigen_values = LA.eigvals(Au)
            sqared_eigen_values = np.sqrt(eigen_values)

            main_vander_matrix = np.vander(eigen_values, increasing=True)
            main_deter_vander = LA.det(main_vander_matrix)

            for i in range(main_vander_matrix.shape[0]):
                vander_matrix_n = np.vander(eigen_values, increasing=True)

                vander_matrix_n[:,i] = sqared_eigen_values
                matrix_deter_n = LA.det(vander_matrix_n)

                list_of_matrices.append(vander_matrix_n)
                list_of_deters.append(matrix_deter_n)

            sample = np.zeros(main_vander_matrix.shape, main_vander_matrix.dtype)
            for i in range(main_vander_matrix.shape[0]):
                term = ((Au**i)*(list_of_deters[i]/main_deter_vander)) # Needs to be checked!
                sample += term

            lambda_U_main_matrix.append(sample)

        ###   Расчет матрицы лямбда тока !!!
        for harm in range(Z.shape[0]):
            list_of_matrices = []
            list_of_deters = []
            Ai = np.dot(Y[harm], Z[harm])
            eigen_values = LA.eigvals(Ai)
            sqared_eigen_values = np.sqrt(eigen_values)

            main_vander_matrix = np.vander(eigen_values, increasing=True)
            main_deter_vander = LA.det(main_vander_matrix)

            for i in range(main_vander_matrix.shape[0]):
                vander_matrix_n = np.vander(eigen_values, increasing=True)

                vander_matrix_n[:,i] = sqared_eigen_values
                matrix_deter_n = LA.det(vander_matrix_n)

                list_of_matrices.append(vander_matrix_n)
                list_of_deters.append(matrix_deter_n)

            sample = np.zeros(main_vander_matrix.shape, main_vander_matrix.dtype)
            for i in range(main_vander_matrix.shape[0]):
                term = ((Ai**i)*(list_of_deters[i]/main_deter_vander)) # Needs to be checked!
                sample += term

            lambda_I_main_matrix.append(sample)

        # Возвращает 2 матрицы размерностью [50x4x4] каждая для лямбды 1)напряжения и лямбды 2)тока соответственно
        return np.array(lambda_U_main_matrix), np.array(lambda_I_main_matrix)



    # Функция для расчета комплексных квадратных матриц экспонен_напряжения и экспонента_ток
    def calculate_exponentials(self, lambda_UI, x):

        # Проверка правильности входных матриц
        assert len(lambda_UI.shape) == 3, "Lambda matrix must be 3 dimentional matrix"
        assert lambda_UI.shape[1] == lambda_UI.shape[2], "Lambda matrix must square matrix"
        assert type(x) in [int, float], "Distance x must be either int or float type"

        positive_expon_matrix = []
        negative_expon_matrix = []

        # Расчет положительной матрицы
        for harm in range(lambda_UI.shape[0]):
            list_of_matrices = []
            list_of_deters = []

            Aui = np.dot(lambda_UI[harm], x)
            eigen_values = LA.eigvals(Aui)
            expon_eigen_values = np.exp(eigen_values)

            main_vander_matrix = np.vander(eigen_values, increasing=True)
            main_deter_vander = LA.det(main_vander_matrix)

            for i in range(main_vander_matrix.shape[0]):
                vander_matrix_n = np.vander(eigen_values, increasing=True)

                vander_matrix_n[:,i] = expon_eigen_values
                matrix_deter_n = LA.det(vander_matrix_n)

                list_of_matrices.append(vander_matrix_n)
                list_of_deters.append(matrix_deter_n)

            sample = np.zeros(main_vander_matrix.shape, main_vander_matrix.dtype)
            for i in range(main_vander_matrix.shape[0]):
                term = ((Aui**i)*(list_of_deters[i]/main_deter_vander)) # Needs to be checked!
                sample += term

            positive_expon_matrix.append(sample)


        # Расчет отрицательной матрицы
        for harm in range(lambda_UI.shape[0]):
            list_of_matrices = []
            list_of_deters = []

            Aui = np.dot(lambda_UI[harm], -1*x)
            eigen_values = LA.eigvals(Aui)
            expon_eigen_values = np.exp(eigen_values)

            main_vander_matrix = np.vander(eigen_values, increasing=True)
            main_deter_vander = LA.det(main_vander_matrix)

            for i in range(main_vander_matrix.shape[0]):
                vander_matrix_n = np.vander(eigen_values, increasing=True)

                vander_matrix_n[:,i] = expon_eigen_values
                matrix_deter_n = LA.det(vander_matrix_n)

                list_of_matrices.append(vander_matrix_n)
                list_of_deters.append(matrix_deter_n)

            sample = np.zeros(main_vander_matrix.shape, main_vander_matrix.dtype)
            for i in range(main_vander_matrix.shape[0]):
                term = ((Aui**i)*(list_of_deters[i]/main_deter_vander)) # Needs to be checked!
                sample += term

            negative_expon_matrix.append(sample)

        # Возвращает 2 матрицы размерностью [50x4x4] каждая для +/- экспоненциальной для определенной длины "х"
        return np.array(positive_expon_matrix), np.array(negative_expon_matrix)



# Функции для специальной парной итерации
def pairwise_1(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def pairwise_2(iterable):
    a = iter(iterable)
    return zip(a, a)


# Функция для специальной итерации через мелкий шаг
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


class LoopI():
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
        self.current_index = self.start
        self.the_end = False

    def step_up_index(self):
        if self.current_index < self.stop:
            self.current_index += 1
            self.the_end = False
            if self.current_index == self.stop:
                self.the_end = True


    def get_current_index(self):
        return self.current_index

    def set_current_index(self, value):
        if value < self.stop:
            self.current_index = value
            self.the_end = False
        elif value == self.stop:
            self.current_index = value
            self.the_end = True
        else:
            self.the_end = True


# Класс для рабочих методов для получения значении с файла
class PhaseHolder():

    volt = None

    def __init__(self, to_put):
        self.df = None
        self.add_df = None
        self.to_put = to_put


    def __call__(self, iu):
        iu = iu.lower()
        if iu == 'u':
            return self.df.iloc[:int(len(self.df.index)/2),:]
        if iu == "i":
            return self.df.iloc[int(len(self.df.index)/2):,:].reset_index(drop=True)


     # Метод для получения значении отдельной гармоники [Начиная со  2-ой]
    def get_harmonic(self):
        check_dict = {"u":4,"i":5}
        phase_df = getattr(self, 'df')
        all_columns = len(phase_df.columns)

        Uphase_df = phase_df.iloc[:int(len(phase_df.index)/2),:]
        Iphase_df = phase_df.iloc[int(len(phase_df.index)/2):,:]

        index_num = len(Uphase_df.index)

        Is_amps = [np.zeros(index_num)]; Is_phas = [np.zeros(index_num)];
        Us_amps = [np.zeros(index_num)]; Us_phas = [np.zeros(index_num)];

        for i, (a, p) in enumerate(pairwise_2(range(all_columns)), start=2): # Начало 2 !!!
            AU = Uphase_df.iloc[:,a].reset_index(drop=True).tolist(); AI = Iphase_df.iloc[:,a].reset_index(drop=True).tolist();
            PU = Uphase_df.iloc[:,p].reset_index(drop=True).tolist(); PI = Iphase_df.iloc[:,p].reset_index(drop=True).tolist();

            Is_amps.append(AI); Is_phas.append(PI); Us_amps.append(AU); Us_phas.append(PU)


        return np.array(Us_amps), np.array(Us_phas), np.array(Is_amps), np.array(Is_phas)



    # Метод для сложения всех гармоник для определенного присоединения я его фазы + значение
    # основного(номинального) напряжения
    def add_harmonics(self, iu):
        check_dict = {"u":4,"i":5}
        SIN = None; iu = iu.lower();
        phase_df = getattr(self, 'df')
        all_columns = len(phase_df.columns)
        if iu == 'u':
            phase_df = phase_df.iloc[:int(len(phase_df.index)/2),:]
        if iu == 'i':
            phase_df = phase_df.iloc[int(len(phase_df.index)/2):,:]
        for a, p in pairwise_2(range(all_columns)):
            A = phase_df.iloc[:,a].reset_index(drop=True) * self.add_df.iloc[:,check_dict[iu]].reset_index(drop=True)/100
            P = phase_df.iloc[:,p].reset_index(drop=True); Pn = P.apply(lambda entry: np.sin(entry*np.pi/180))

            if type(SIN)==type(None):
                SIN = A * Pn
            else:
                SIN += A * Pn
        return SIN.reset_index(drop=True)


    def get_rms(self):
        # returns rmsu and rmsi
        thd_u, thd_i = self.add_df.iloc[:,0].reset_index(drop=True).tolist(), self.add_df.iloc[:,1].reset_index(drop=True).tolist()
        rmsu, rmsi = self.add_df.iloc[:,2].reset_index(drop=True).tolist(), self.add_df.iloc[:,3].reset_index(drop=True).tolist()
        return np.array(thd_u), np.array(thd_i), np.array(rmsu), np.array(rmsi)


    def get_main_harm(self):

        Ua = 4; Up = 6;
        Ia = 5; Ip = 7;

        AU = self.add_df.iloc[:,Ua].reset_index(drop=True).tolist(); AI = self.add_df.iloc[:,Ia].reset_index(drop=True).tolist();
        PU = self.add_df.iloc[:,Up].reset_index(drop=True).tolist(); PI = self.add_df.iloc[:,Ip].reset_index(drop=True).tolist();

        return np.array(AU), np.array(PU), np.array(AI), np.array(PI)


# Класс для инициализации основных атрибутов для присоединении
class AttribHolder():

    def __init__(self):
        self.naimen=None

        self.faza_A=PhaseHolder(None)
        self.faza_B=PhaseHolder(None)
        self.faza_C=PhaseHolder(None)

        self.faza_A2=PhaseHolder(None)
        self.faza_B2=PhaseHolder(None)
        self.faza_C2=PhaseHolder(None)



# Класс для чтения файла EXCEL и предобработки имеющихся там данных
class PodStans():

    check = type(None)
    """
    В объекте данного класса будут следующие атрибуты:
        - .vse_prisoed --> возвращает список всех присоединении для данной подстанции
        - .prisoed_n   --> n - номер присоединения. n-ое количество атрибутов для каждого
                            присоединения. (Назначается автоматически по мере чтения Excel файла)

                далее, вышеуказанный атрибут(-ы), в свою очередь, имеет(-ют) в себе атрибуты:
                    - .naimen   --> наименование присоединения
                    - .faza_A   --> гармоники для фазы А
                    - .faza_B   --> гармоники для фазы В
                    - .faza_C   --> гармоники для фазы С

                             а каждый атрибут фазы имеет следующие методы:
                                - .get_harmonic(<harmonic>, <iu>)
                                        harmonic - Номер гармоники [2-50]
                                        iu - Ток или Напряжение ["U" или "I"]
                                - .add_harmonics(<iu>)
                                        iu - Ток или Напряжение ["U" или "I"]
                                - .get_rms(<iu>)
                                        iu - Ток или Напряжение ["U" или "I"]
                                - .get_main_harm(<iu>)
                                        iu - Ток или Напряжение ["U" или "I"]
                                - каждый артрибут фазы может быть вызван при помощи ("U") или ("i")
                                        для возвращения значении напряжения или тока для выбранной фазы

   """

    # Инициализация файла и рабочих атрибутов. Вызов создающих методов
    def __init__(self, path, *,volt=110, harm_num=49, t_izmer=560):
        self.file_path = path
        self.harm_num = harm_num
        self.t_izmer = t_izmer
        self.excel_file = pd.ExcelFile(path)
        self.sheet_names = self.excel_file.sheet_names
        self.label_started = dict()
        self.get_prisoed_labels(self.excel_file, self.label_started)
        self.create_arrange_prisoed()
        PhaseHolder.volt = volt*10**3
        self.vse_prisoed = list(self.label_started.keys())


    # Метод для выведения количества и присоединении в файле
    def get_prisoed_labels(self, excel_file, label_started):

        dataframe = pd.read_excel(excel_file, excel_file.sheet_names[0], header=None)
        for i, row in enumerate(dataframe.iloc[:,0]):
            if type(row) == str:
                label_started[row] = i
            else:
                continue

        for i, item in enumerate(label_started.keys()):
            label_started[item] -= i


    # Метод для добавления таблицы гармоник
    def add_dataframes(self, dataframe, phase):

        for pris_num, chunk in enumerate(pairwise_1(list(self.label_started.values())+[None]), start=1):
            nach, kon = chunk
            cleaned_df = dataframe.dropna(thresh=50)

            faza_new = PhaseHolder(cleaned_df.iloc[nach:kon, :self.harm_num*2])

            pris = getattr(self, "prisoed_"+str(pris_num))
            faza = getattr(pris, "faza_"+str(phase))
            faza_df = getattr(faza, "df")
            if type(faza_df) == type(None):
                setattr(faza, "df", faza_new.to_put)
            else:
                appended_df = faza_df.append(faza_new.to_put, ignore_index=True)
                setattr(faza, "df", appended_df)

            add_df = PhaseHolder(cleaned_df.iloc[nach:kon, self.harm_num*2:])

            if len(add_df.to_put[add_df.to_put.notna()].columns) == 8:
                rest_df = getattr(faza, "add_df")
                if type(rest_df) == type(None):
                    setattr(faza, "add_df", add_df.to_put)
                else:
                    appended_df = rest_df.append(add_df.to_put, ignore_index=True)
                    setattr(faza, "add_df", appended_df)



    # Метод для заполнения атрибутов данными
    def create_arrange_prisoed(self):

        for i, pris in enumerate(self.label_started.values(), start=1):
            attr_name = "prisoed_" + str(i)
            setattr(self, attr_name, AttribHolder())

        for i, name in enumerate(self.label_started.keys(), start=1):
            pris = getattr(self, "prisoed_"+str(i))
            setattr(pris, "naimen", name)

        for i, sheet in enumerate(self.sheet_names, start=1):
            dataframe = pd.read_excel(self.excel_file, sheet, header=None)


            if i > 10:
                self.add_dataframes(dataframe, "C2")
            elif i > 8:
                self.add_dataframes(dataframe, "B2")
            elif i > 6:
                self.add_dataframes(dataframe, "A2")
            elif i > 4:
                self.add_dataframes(dataframe, "C")
            elif i > 2:
                self.add_dataframes(dataframe, "B")
            else:
                self.add_dataframes(dataframe, "A")



# Implementing other FORTRAN's subroutines
def DLINRG(A):
    """
    Computes the inverse of a real general matrix.

        N - order of the matrix (rows x columns). (Input)
        A - N by N matrix containing the matrix to be inverted. (Input)
        DA — Leading dimension of A exactly as specified in the dimension statement of the calling program. (Input)
        AINV — N by N matrix containing the inverse of A. (Output)
        If A is not needed, A and AINV can share the same storage locations.
        LDAINV — Leading dimension of AINV exactly as specified in the dimension statement of the calling program. (Input)
    """
    return np.linalg.inv(A)


def DEVLCG(A):
    """
    Compute all of the eigenvalues of a complex matrix.

        N — Order of the matrix A.   (Input)
        A — Complex matrix of order N.   (Input)
        LDA — Leading dimension of A exactly as specified in the dimension statement in the calling program.   (Input)
        EVAL —  Complex vector of length N containing the eigenvalues of A in decreasing order of magnitude.   (Output)
    """
    return LA.eigvals(A)


def DLFTCG(A):
    """
    Compute the LU factorization of a complex general matrix.

        N — Order of the matrix.   (Input)
        A — Complex N by N matrix to be factored.   (Input)
        LDA — Leading dimension of A exactly as specified in the dimension statement of the calling program.   (Input)
        FAC — Complex N by N matrix containing the LU factorization of the matrix A.   (Output)
        If A is not needed, A and FAC can share the same storage locations.
        LDFAC — Leading dimension of FAC exactly as specified in the dimension statement of the calling program.   (Input)
        IPVT — Vector of length N containing the pivoting information for the LU factorization.   (Output)
    """
    return lu_factor(A)[0]


def LFSCG(FAC, IPVT, B):
    """
     Solve a complex general system of linear equations given the LU factorization of the coefficient matrix.

        N — Number of equations.   (Input)
        FAC — Complex N by N matrix containing the LU factorization of the coefficient matrix A as output from routine LFCCG/DLFCCG or LFTCG/DLFTCG.   (Input)
        LDFAC — Leading dimension of FAC exactly as specified in the dimension statement of the calling program.   (Input)
        IPVT — Vector of length N containing the pivoting information for the LU factorization of A as output from routine LFCCG/DLFCCG or LFTCG/DLFTCG.   (Input)
        B — Complex vector of length N containing the right-hand side of the linear system.   (Input)
        IPATH — Path indicator.   (Input)
        IPATH = 1 means the system AX = B is solved.
        IPATH = 2 means the system AHX = B is solved.
        X — Complex vector of length N containing the solution to the linear system.   (Output)
        If B is not needed, B and X can share the same storage locations.
    """
    return lu_solve((FAC, IPVT), B)


def DLFDCG(A):
    """ !!! EXCEPTION ACCEPT ONLY A !!!
    Compute the determinant of a complex general matrix given the LU factorization of the matrix.

        N — Order of the matrix.   (Input)
        FAC — Complex N by N matrix containing the LU factorization of the matrix A as output from routine LFTCG/DLFTCG or LFCCG/DLFCCG.   (Input)
        LDFAC — Leading dimension of FAC exactly as specified in the dimension statement of the calling program.   (Input)
        IPVT — Vector of length N containing the pivoting information for the LU factorization as output from routine LFTCG/DLFTCG or LFCCG/DLFCCG.   (Input)
        DET1 — Complex scalar containing the mantissa of the determinant.   (Output)
        The value DET1 is normalized so that 1.0 £ |DET1| < 10.0 or DET1 = 0.0.
        DET2 — Scalar containing the exponent of the determinant.   (Output)
        The determinant is returned in the form det(A) = DET1 * 10DET2.
    """
    return np.linalg.det(A)


def DMRRRR(A, B):
    """
    Multiply two real rectangular matrices, AB.

        NRA — Number of rows of A.   (Input)
        NCA — Number of columns of A.   (Input)
        A — Real NRA by NCA matrix in full storage mode.   (Input)
        LDA — Leading dimension of A exactly as specified in the dimension statement of the calling program.   (Input)
        NRB — Number of rows of B.   (Input)
        NRB must be equal to NCA.
        NCB — Number of columns of B.   (Input)
        B — Real NRB by NCB matrix in full storage mode.   (Input)
        LDB — Leading dimension of B exactly as specified in the dimension statement of the calling program.   (Input)
        NRC — Number of rows of C.   (Input)
        NRC must be equal to NRA.
        NCC — Number of columns of C.   (Input)
        NCC must be equal to NCB.
        C — Real NRC by NCC matrix containing the product AB in full storage mode.   (Output)
        LDC — Leading dimension of C exactly as specified in the dimension statement of the calling program.   (Input)
    """
    return np.matmul(A, B)


def DMUCRV(A, X):
    """
    Multiply a complex rectangular matrix by a complex vector.

        NRA — Number of rows of A.   (Input)
        NCA — Number of columns of A.   (Input)
        A — Complex NRA by NCA rectangular matrix.   (Input)
        LDA — Leading dimension of A exactly as specified in the dimension statement of the calling program.   (Input)
        NX — Length of the vector X.   (Input)
        NX must be equal to NCA if IPATH is equal to 1. NX must be equal to NRA if IPATH is equal to 2.
        X — Complex vector of length NX.   (Input)
        IPATH — Integer flag.   (Input)
        IPATH = 1 means the product Y = A * X is computed. IPATH = 2 means the product Y = trans(A) * X is computed, where trans(A) is the transpose of A.
        NY — Length of the vector Y.   (Input)
        NY must be equal to NRA if IPATH is equal to 1. NY must be equal to NCA if IPATH is equal to 2.
        Y — Complex vector of length NY containing the product A * X if IPATH is equal to 1 and the product trans(A) * X if IPATH is equal to 2.   (Output)
    """
    return np.matmul(A, X)


def DMCRCR(A, B):
    """
    Multiply two complex rectangular matrices, AB.

        NRA — Number of rows of A.   (Input)
        NCA — Number of columns of A.   (Input)
        A — Complex NRA by NCA rectangular matrix.   (Input)
        LDA — Leading dimension of A exactly as specified in the dimension statement of the calling program.   (Input)
        NRB — Number of rows of B.   (Input)
        NRB must be equal to NCA.
        NCB — Number of columns of B.   (Input)
        B — Complex NRB by NCB rectangular matrix.   (Input)
        LDB — Leading dimension of B exactly as specified in the dimension statement of the calling program.   (Input)
        NRC — Number of rows of C.   (Input)
        NRC must be equal to NRA.
        NCC — Number of columns of C.   (Input)
        NCC must be equal to NCB.
        C — Complex NRC by NCC rectangular matrix containing the product A * B.   (Output)
        LDC — Leading dimension of C exactly as specified in the dimension statement of the calling program.   (Input)
    """
    return np.matmul(A, B)


def DLSLCG(A, X):
    """
    Solve a complex general system of linear equations without iterative refinement.

        N — Number of equations.   (Input)
        A — Complex N by N matrix containing the coefficients of the linear system.   (Input)
        LDA — Leading dimension of A exactly as specified in the dimension statement of the calling program.   (Input)
        B — Complex vector of length N containing the right-hand side of the linear system.   (Input)
        IPATH — Path indicator.   (Input)
        IPATH = 1 means the system AX = B is solved.
        IPATH = 2 means the system AHX = B is solved.
        X — Complex vector of length N containing the solution to the linear system.   (Output)
        If B is not needed, B and X can share the same storage locations.
    """
    try:
        OUT = np.linalg.inv(A).dot(X)
    except:
        OUT = np.linalg.pinv(A).dot(X)
    return OUT


def DLINCG(N, A):
    """ !!! SCALE FIRST TO 1/N !!!
    Compute the inverse of a complex general matrix.

        N — Order of the matrix A.   (Input)
        A — Complex N by N matrix containing the matrix to be inverted.   (Input)
        LDA — Leading dimension of A exactly as specified in the dimension statement of the calling program.   (Input)
        AINV — Complex N by N matrix containing the inverse of A.   (Output)
        If A is not needed, A and AINV can share the same storage locations.
        LDAINV — Leading dimension of AINV exactly as specified in the dimension statement of the calling program.   (Input)
    """
    # if N == True:
    #     A = A*1/N

    try:
        OUT = np.linalg.inv(A)
    except:
        OUT = np.linalg.pinv(A)

    return OUT
