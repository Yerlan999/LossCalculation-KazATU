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

        Is_amps = []; Is_phas = [];
        Us_amps = []; Us_phas = [];

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



# Implementing RASCHET fortran's subroutine in python
# Implementing RASCHET fortran's subroutine in python
def RASCHET():
    global UK1,AIK1,LL,NN,PPP,PP1,PP2,PPP1,PPP2,PPP3,PPP4,PPP5,PPP6,PPP7,PPP8, MM,M,M1,MT,M10,M20,PR,K1,K2,K3,N1,N2,N3,MPR,MTR,MMT

    XA, YA, UXM, GM, OMP, R0, R, S, HI, R11, DET2, DET4 = np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M))
    EVU, B, UX, AIX, UK1, AIK1, SM, EVI, DET1, DET3, BB, AIXM, AA = np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)), np.zeros((M)),
    B1, B4, B5 = np.zeros((M2)), np.zeros((M2)), np.zeros((M2))
    B6, B10, B7 = np.zeros((M10)), np.zeros((M10)), np.zeros((M10))

    DET10, DET20, SS, SS1, EX1,  = [], [], [], [], []

    HC1, HC2, HC3, HC4, F10, XL, XL1, G, D, HC, UXM = np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M))
    Z, Y, AU, E, F, AAI, F1 = np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M))
    F2, D1, D2, D3, LU, LI = np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M))
    LU1, LI1, LU2, LU3, LI2, LI3 = np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M))
    F3, F4, F5, F6, F7 = np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M))
    HH13, HH14, HH21, HH22, HH23 = np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M))
    HH31, HH32, HH33, HH34, HH24 = np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M))
    HH41, HH42, HH43, HH44 = np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M))
    CC, DD, HH11, HH12 = np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M))

    AG, GG = np.zeros((M1, M, M)), np.zeros((M20, M20))
    GG1, GG2, GG3, GG4, GG5 = np.zeros((M20, M20)), np.zeros((M20, M20)), np.zeros((M10, M20)), np.zeros((M10, M10)), np.zeros((M10, M10))
    IPVT1, IH = np.zeros((M1)), np.zeros((M20))

    if PR == 1:
        PP1 = 0
    if PR == 2:
        PP2 = 0
    PPP[NN, LL] = 0
    PPP1[NN, LL] = 0
    PPP2[NN, LL] = 0
    PPP3[NN, LL] = 0
    PPP4[NN, LL] = 0
    PPP5[NN, LL] = 0
    PPP6[NN, LL] = 0
    PPP7[NN, LL] = 0
    PPP8[NN, LL] = 0

    MMT = MM / MT
    W = float(LL)
    EX1 = complex(2.71828, 0)

    # WRITING INTO UNIT 5. LINE MAIN CHARACTERISTIC
    for i in range(M):
        R[i]=np.sqrt(S[i]/np.pi)/1000.
        HI[i]=R[i]/(2.)*np.sqrt(2*np.pi*W*50*4*np.pi*OMP[i]*GM[i]/20)
        R0[i]=1000/(GM[i]*S[i])
        if HI[i] < 1:
            R11[i]=R0[i]*(1+HI[i]**4/3)
        if HI[i] > 1:
            R11[i]=R0(i)*(HI[i]+0.25+3./(64.*HI[i]))
        if i+1 == M:
            pass
            # WRITE R11's values INTO THE FILE "R11"

    # 845 - 12
    for i in range(M10):
        for j in range(M10):
            HH[i,j] = 0

    # 12 - 161
    for i in range(M):
        for j in range(M):
            if i == j:
                D[i,i] = R[i]
            if i != j:
                D[i,j]=np.sqrt((XA[i]-XA[j])**2+(YA[i]-YA[j])**2)
            HC[i,j]=np.sqrt((XA[i]-XA[j])**2+(YA[i]+YA[j])**2)
            E[i,j]=complex(0.0, 0.0)
            E[i,i]=complex(1.0, 0.0)

    # 161 - 740
    for i in range(M):
        for j in range(M):
            XL1[i,j]=0.145*np.log10(1000/D[i,j])/np.pi

    # 740 - 743
    for i in range(M):
        for j in range(M):
            HC1[i,j]=41.4*10.**6*np.log10(HC[i,j]/D[i,j])

    # 743 - 744
    HC3 = DLINRG(M,HC1,M,HC3,M)
    F10 = DMRRRR(M,M,HC1,M,M,M,HC3,M,M,M,F10,M)

    for i in range(M):
        for j in range(M):
            HC2[i,j]=HC3[i,j]*2.*np.pi*50

    # 744 - 847
    for i in range(M):
        for j in range(M):
            XL[i,j]=XL1[i,j]*W*2*50*np.pi
            HC4[i,j]=HC2[i,j]*W
            R10=0.0
            if i == j:
                Z[i,j]=complex(R11[i],XL[i,j])
            if i != j:
                Z[i,j]=complex(R10,XL[i,j])
            if i == j:
                G[i,j]=0.00000004*YA[i]/YA[i]
            if i != j:
                G[i,j]=-0.00000004*YA[1]/D[i,j]
            G[i,j]=0.
            Y[i,j]=complex(G[i,j],HC4[i,j])


    # 847 - the end of subroutine
    for III in range(MT): # <----- main loop # 1300

        if M != 3: # GOTO 767
            if (M != 4): # GOTO 768
                if (M != 6): # GOTO 769
                    if (M != 7): # GOTO 770
                        if (M != 8): # GOTO 771
                            AU = DMCRCR(M,M,Z,M,M,M,Y,M,M,M,AU,M)
                            SS1=np.sqrt(AU[1,1])
                            EVU = DEVLCG(M,AU,M,EVU)
                            # IMPORTANT POINT
                            for j in range(M):
                                for i in range(M):
                                    F[i,j]=EVU[i]**(j) # (j-1) was in the first case
                                    F1[i,j]=F[i,j]

                            for i in range(M1):
                                for j in range(M1):
                                    A1[i,j]=F[i,j]

                            A2 = DLFTCG(M1,A1,M1,A2,M1,IPVT1)
                            SS = DET10 = DLFDCG(M1,A2,M1,IPVT1); DET20 = 1 # parameters DET10,DET20 were in function call
                            # SS=DET10*(10.**DET20)

                            for j in range(M1):

                                for ii in range(M1):
                                    for jj in range(M1):
                                        F[ii,jj]=F1[ii,jj]

                                for i in range(M1):
                                    F[i,j]=EVU[i]**0.5

                                for ii in range(M1):
                                    for jj in range(M1):
                                        A1[ii,jj]=F[ii,jj]

                                A2 = DLFTCG(M1,A1,M1,A2,M1,IPVT1)
                                DET1[j] = DLFDCG(M1,A2,M1,IPVT1); DET2[j] = 1 # parameters DET1(J),DET2(J) were in function call


                            F2 = DMCRCR(M,M,AU, M,M,M,AU,M,M,M,F2,M)
                            F3 = DMCRCR(M,M,F2,M,M,M,AU,M,M,M,F3,M)
                            F4 = DMCRCR(M,M,F3,M,M,M,AU,M,M,M,F4,M)
                            F5 = DMCRCR(M,M,F4,M,M,M,AU,M,M,M,F5,M)
                            F6 = DMCRCR(M,M,F5,M,M,M,AU,M,M,M,F6,M)
                            F7 = DMCRCR(M,M,F6,M,M,M,AU,M,M,M,F7,M)

                            for i in range(M1):
                                for ii in range(M):
                                    for jj in range(M):

                                        if i == 1:
                                            AG[1,ii,jj]=E[ii,jj]
                                        if i == 2:
                                            AG[2,ii,jj]=AU[ii,jj]
                                        if i == 3:
                                            AG[3,ii,jj]=F2[ii,jj]
                                        if i == 4:
                                            AG[4,ii,jj]=F3[ii,jj]
                                        if i == 5:
                                            AG[5,ii,jj]=F4[ii,jj]
                                        if i == 6:
                                            AG[6,ii,jj]=F5[ii,jj]
                                        if i == 7:
                                            AG[7,ii,jj]=F6[ii,jj]
                                        if i == 8:
                                            AG[8,ii,jj]=F7[ii,jj]

                            for ii in range(M):
                                for jj in range(M):
                                    LU[ii,jj]=0.

                            for i in range(M1):
                                DET3[i]=DET1[i]/DET10
                                DET4[i]=DET2[i]-DET20
                                for ii in range(M):
                                    for jj in range(M):
                                        LU[ii,jj]=LU[ii,jj]+AG[i,ii,jj]*DET3[i]*(10**DET4[i])


                            F3 = DMCRCR(M,M,LU,M,M,M,LU,M,M,M,F3,M)
                            AAI = DMCRCR(M,M,Y,M,M,M,Z,M,M,M,AAI,M)
                            EVI = DEVLCG(M,AAI,M,EVI)

                            for j in range(M):
                                for i in range(M):
                                    F[i,j] = EVI[i]**[j-1]
                                    F1[i,j] = F[i,j]

                            for i in range(M1):
                                for j in range(M1):
                                    A1[i,j]=F[i,j]

                            A2 = DLFTCG(M1,A1,M1,A2,M1,IPVT1)
                            DET10 = DLFDCG(M1,A2,M1,IPVT1); DET20 = 1 # parameters DET10,DET20 were in function call

                            for j in range(M1):
                                for ii in range(M1):
                                    for jj in range(M1):
                                        F[ii,jj]=F1[ii,jj]

                                for i in range(M1):
                                    F[i,j]=EVI[i]**0.5
                                for ii in range(M1):
                                    for jj in range(M1):
                                        A1[ii,jj]=F[ii,jj]

                                A2 = DLFTCG(M1,A1,M1,A2,M1,IPVT1)
                                DET1[j] = DLFDCG(M1,A2,M1,IPVT1); DET2[j] = 1 # parameters DET1(J),DET2(J) were in function call

                            F2 = DMCRCR(M,M,AAI, M,M,M,AAI,M,M,M,F2,M)
                            F3 = DMCRCR(M,M,F2,M,M,M,AAI,M,M,M,F3,M)
                            F4 = DMCRCR(M,M,F3,M,M,M,AAI,M,M,M,F4,M)
                            F5 = DMCRCR(M,M,F4,M,M,M,AAI,M,M,M,F5,M)
                            F6 = DMCRCR(M,M,F5,M,M,M,AAI,M,M,M,F6,M)
                            F7 = DMCRCR(M,M,F6,M,M,M,AAI,M,M,M,F7,M)

                            for i in range(M1):
                                for ii in range(M):
                                    for jj in range(M):

                                        if I == 1:
                                            AG[1,ii,jj]=E[ii,jj]
                                        if I == 2:
                                            AG[2,ii,jj]=AAI[ii,jj]
                                        if I == 3:
                                            AG[3,ii,jj]=F2[ii,jj]
                                        if I == 4:
                                            AG[4,ii,jj]=F3[ii,jj]
                                        if I == 5:
                                            AG[5,ii,jj]=F4[ii,jj]
                                        if I == 6:
                                            AG[6,ii,jj]=F5[ii,jj]
                                        if I == 7:
                                            AG[7,ii,jj]=F6[ii,jj]
                                        if I == 8:
                                            AG[8,ii,jj]=F7[ii,jj]

                            for ii in range(M):
                                for jj in range(M):
                                    LI[ii,jj]=0.

                            for i in range(M1):
                               for ii in range(M):
                                    for jj in range(M):
                                        LI[ii,jj]=LI[ii,jj]+(DET1[i]*(10**DET2[i])*AG[i,ii,jj])/(DET10*(10**DET20))


                            F3 = DMCRCR(M,M,LI,M,M,M,LI,M,M,M,F3,M)
                            LM=MMT

                            for n, _ in enumerate(range(2), start=1):

                                if n == 1:
                                    SA=-1.
                                if n == 2:
                                    SA=1.

                                for ii in range(M):
                                    for jj in range(M):
                                        LU1[ii,jj]=SA*MMT*LU[ii,jj]


                                EVU = DEVLCG(M,LU1,M,EVU)
                                for i in rane(M):
                                    for j in range(M):
                                        F[i,j]=EVU[i]**(j) # (j-1) here was an expression (j-1)


                                for ii in range(M):
                                    for jj in range(M):
                                        F1[ii,jj]=F[ii,jj]


                                for ii in range(M1):
                                    for jj in range(M1):
                                        A1[ii,jj]=F[ii,jj]


                                A2 = DLFTCG(M1,A1,M1,A2,M1,IPVT1)
                                DET10 = DLFDCG(M1,A2,M1,IPVT1); DET20 = 1 # DET10 and DET20 were the arguments of this function

                                for j in range(M):
                                    for ii in range(M):
                                        for jj in range(M):
                                            F[ii,jj]=F1[ii,jj]


                                    for i in range(M):
                                        F[i,j]=EX1**EVU[i]


                                    for ii in range(M1):
                                        for jj in range(M2):
                                            A1[ii,jj]=F[ii,jj]


                                    A2 = DLFTCG(M1,A1,M1,A2,M1,IPVT1)
                                    DET1[j] = DLFDCG(M1,A2,M1,IPVT1); DET2[j] = 1


                                F2 = DMCRCR(M,M,LU1, M,M,M,LU1,M,M,M,F2,M)
                                F3 = DMCRCR(M,M,F2,M,M,M,LU1,M,M,M,F3,M)
                                F4 = DMCRCR(M,M,F3,M,M,M,LU1,M,M,M,F4,M)
                                F5 = DMCRCR(M,M,F4,M,M,M,LU1,M,M,M,F5,M)
                                F6 = DMCRCR(M,M,F5,M,M,M,LU1,M,M,M,F6,M)
                                F7 = DMCRCR(M,M,F6,M,M,M,LU1,M,M,M,F7,M)

                                for i in range(M1):
                                    for ii in range(M):
                                        for jj in range(M):

                                            if i == 1:
                                                AG[1,ii,jj]=E[ii,jj]
                                            if i == 2:
                                                AG[2,ii,jj]=LU1[ii,jj]
                                            if i == 3:
                                                AG[3,ii,jj]=F2[ii,jj]
                                            if i == 4:
                                                AG[4,ii,jj]=F3[ii,jj]
                                            if i == 5:
                                                AG[5,ii,jj]=F4[ii,jj]
                                            if i == 6:
                                                AG[6,ii,jj]=F5[ii,jj]
                                            if i == 7:
                                                AG[7,ii,jj]=F6[ii,jj]
                                            if i == 8:
                                                AG[8,ii,jj]=F7[ii,jj]

                                for ii in range(M):
                                    for jj in range(M):
                                        LU2[ii,jj]=0.

                                for i in range(M1):
                                    for ii in range(M):
                                        for jj in range(M):
                                            LU2[ii,jj]=LU2[ii,jj]+(DET1[i]*(10**DET2[i])*AG[i,ii,jj])/(DET10*(10**DET20))

                                if N == 2:
                                    F2 = DMCRCR(M,M,LU3,M,M,M,LU2,M,M,M,F2,M)
                                    continue
                                else:
                                    for ii in range(M):
                                        for jj in range(M):
                                            LU3[ii,jj]=LU2[ii,jj]

                        else:
                            # 770 - 766
                            for i in range(3):
                                B5[i]=UK1[i]
                                B5[i+3]=UK1[i]
                                B5[MPR+1]=complex(0.,0.)
                                B5[MPR+2]=complex(0.,0.)
                                B5[i+M]=AIK1[i]
                                B5[i+M+3]=AIK1[i]
                                B5[2*M-1]=complex(0.,0.)
                                B5[2*M]=complex(0.,0.)
                                B5[i+2*M]=complex(0.,0.)
                                B5[i+2*M+3]=complex(0.,0.)
                                B5[3*M-1]=complex(0.,0.)
                                B5[3*M]=complex(0.,0.)
                                B5[i+3*M]=complex(0.,0.)
                                B5[i+3*M+3]=complex(0.,0.)
                                B5[4*M-1]=complex(0.,0.)
                                B5[4*M]=complex(0.,0.)

                    else:
                        # 769 - 765
                        for i in range(3):
                            B5[i]=UK1[i]
                            B5[i+3]=UK1[i]
                            B5[M]=complex(0.,0.)
                            B5[i+M]=AIK1[i]
                            B5[i+M+3]=AIK1[i]
                            B5[2*M]=complex(0.,0.)
                            B5[i+2*M]=complex(0.,0.)
                            B5[i+2*M+3]=complex(0.,0.)
                            B5[3*M]=complex(0.,0.)
                            B5[i+3*M]=complex(0.,0.)
                            B5[i+3*M+3]=complex(0.,0.)
                            B5[4*M]=complex(0.,0.)


                else:
                    # 768 - 764
                    for i in range(3):
                        B5[i]=UK1[i]
                        B5[i+3]=UK1[i]
                        B5[i+M]=AIK1[i]
                        B5[i+M+3]=AIK1[i]
                        B5[i+2*M]=complex(0.,0.)
                        B5[i+2*M+3]=complex(0.,0.)
                        B5[i+3*M]=complex(0.,0.)
                        B5[i+3*M+3]=complex(0.,0.)

            else:
                # 767 -762
                for i in range(3):
                    B5[i]=UK1[i]
                    B5[M]=complex(0.,0.)
                    B5[i+M]=AIK1[i]
                    B5[2*M]=complex(0.,0.)
                    B5[i+2*M]=complex(0.,0.)
                    B5[3*M]=complex(0.,0.)
                    B5[i+3*M]=complex(0.,0.)
                    B5[4*M]=complex(0.,0.)

        else:
            # 847 - 761
            for i in range(3):
                B5[i]=UK1[i]
                B5[i+3]=AIK1[i]
                B5[i+6]=complex(0.,0.)
                B5[i+9]=complex(0.,0.)

    for n in range(2):
        if n == 1:
            SA=-1
        if n == 2:
            SA=1

        for ii in range(M):
            for jj in range(M):
                LI1[ii,jj]=SA*MMT*LI[ii,jj]

        EVI = DEVLCG(M,LI1,M,EVI)

        for i in range(M):
            for j in range(M):
                F[i,j]=EVI[i]**(J) # J-1 in original

        for ii in range(M):
            for jj in range(M):
                F1[ii,jj]=F[ii,jj]

        for ii in range(M1):
            for jj in range(M1):
                A1[ii,jj]=F[ii,jj]

        A2 = DLFTCG(M1,A1,M1,A2,M1,IPVT1)
        DET10 = DLFDCG(M1,A2,M1,IPVT1); DET20 = 1

        for j in range(M):
            for ii in range(M):
                for jj in range(M):
                    F[Iii,jj]=F1[ii,jj]

            for i in range(M):
                F[i,j]=EX1**EVI[i]

            for ii in range(M1):
                for jj in range(M1):
                    A1[ii,jj]=F[ii,jj]

            A2 = DLFTCG(M1,A1,M1,A2,M1,IPVT1)
            DET1[j] = DLFDCG(M1,A2,M1,IPVT1); DET2[j] = 1


        F2 = DMCRCR(M,M,LI1, M,M,M,LI1,M,M,M,F2,M)
        F3 = DMCRCR(M,M,F2,M,M,M,LI1,M,M,M,F3,M)
        F4 = DMCRCR(M,M,F3,M,M,M,LI1,M,M,M,F4,M)
        F5 = DMCRCR(M,M,F4,M,M,M,LI1,M,M,M,F5,M)
        F6 = DMCRCR(M,M,F5,M,M,M,LI1,M,M,M,F6,M)
        F7 = DMCRCR(M,M,F6,M,M,M,LI1,M,M,M,F7,M)

        for i in range(M1):
            for ii in range(M):
                for jj in range(M):
                    if i == 1:
                        AG[1,ii,jj]=E[ii,jj]
                    if i == 2:
                        AG[2,ii,jj]=LI1[ii,jj]
                    if i == 3:
                        AG[3,ii,jj]=F2[ii,jj]
                    if i == 4:
                        AG[4,ii,jj]=F3[ii,jj]
                    if i == 5:
                        AG[5,ii,jj]=F4[ii,jj]
                    if i == 6:
                        AG[6,ii,jj]=F5[ii,jj]
                    if i == 7:
                        AG[7,ii,jj]=F6[ii,jj]
                    if i == 8:
                        AG[8,ii,jj]=F7[ii,jj]

        for ii in range(M):
            for jj in range(M):
                LI2[ii,jj]=0

        for i in range(M1):
            for ii in range(M):
                for jj in range(M):
                    LI2[ii,jj]=LI2[ii,jj]+(DET1[i]*(10**DET2[i])*AG[i,ii,jj])/(DET10*(10**DET20))

        if n == 2:
            continue
        else:
            for ii in range(M):
                for jj in range(M):
                    LI3[ii,jj]=LI2[ii,jj]


    F2 = DMCRCR(M,M,LI3, M,M,M,LI2,M,M,M,F2,M)

    for i in range(M20):
        for j in range(M20):
            GG[i,j]=0.
            GG1[i,j]=0.
            GG2[i,j]=0.

    for i in range(M):
        GG[i,i]=1.
        GG[i,i+M]=1.
        GG[i+M,i+2*M]=1.
        GG[i+M,i+3*M]=1.
        GG1[i,i]=1.
        GG1[i,i+M]=1.
        GG1[i+M,i+2*M]=1.
        GG1[i+M,i+3*M]=1.

    for i in range(M):
        for j in range(M):
            GG[i+2*M,j]=-LU[i,j]
            GG[i+2*M,j+M]=LU[i,j]
            GG[i+3*M,j+2*M]=-LI[i,j]
            GG[i+3*M,j+3*M]=+LI[i,j]
            GG1[i+2*M,j]=LU3[i,j]
            GG1[i+2*M,j+M]=LU2[i,j]
            GG1[i+3*M,j+2*M]=LI3[i,j]
            GG1[i+3*M,j+3*M]=LI2[i,j]

    GG2 = DLINCG (M20,GG1,M20,GG2,M20) # Look for M20

    for i in range(M):
        for j in range(M):
            HH11[i,j]=GG2[i,j]
            HH12[i,j]=GG2[i,j+M]
            HH13[i,j]=GG2[i,j+2*M]
            HH14[i,j]=GG2[i,j+3*M]
            HH21[i,j]=GG2[i+M,j]
            HH22[i,j]=GG2[i+M,j+M]
            HH23[i,j]=GG2[i+M,j+2*M]
            HH24[i,j]=GG2[i+M,j+3*M]
            HH31[i,j]=GG2[i+2*M,j]
            HH32[i,j]=GG2[i+2*M,j+M]
            HH33[i,j]=GG2[i+2*M,j+2*M]
            HH34[i,j]=GG2[i+2*M,j+3*M]
            HH41[i,j]=GG2[i+3*M,j]
            HH42[i,j]=GG2[i+3*M,j+M]
            HH43[i,j]=GG2[i+3*M,j+2*M]
            HH44[i,j]=GG2[i+3*M,j+3*M]

    F = DMCRCR(M,M,LU3,M,M,M,HH11,M,M,M,F,M)
    HH11 = DMCRCR(M,M,LU,M,M,M,F,M,M,M,HH11,M)
    F = DMCRCR(M,M,LU2,M,M,M,HH21,M,M,M,F,M)
    HH21 = DMCRCR(M,M,LU,M,M,M,F,M,M,M,HH21,M)
    F = DMCRCR(M,M,LU3,M,M,M,HH13,M,M,M,F,M)
    HH13 = DMCRCR(M,M,LU,M,M,M,F,M,M,M,HH13,M)
    F = DMCRCR(M,M,LU2,M,M,M,HH23,M,M,M,F,M)
    HH23 = DMCRCR(M,M,LU,M,M,M,F,M,M,M,HH23,M)
    F = DMCRCR(M,M,LI3,M,M,M,HH32,M,M,M,F,M)
    HH32 = DMCRCR(M,M,LI,M,M,M,F,M,M,M,HH32,M)
    F = DMCRCR(M,M,LI2,M,M,M,HH42,M,M,M,F,M)
    HH42 = DMCRCR(M,M,LI,M,M,M,F,M,M,M,HH42,M)
    F = DMCRCR(M,M,LI3,M,M,M,HH34,M,M,M,F,M)
    HH34 = DMCRCR(M,M,LI,M,M,M,F,M,M,M,HH34,M)
    F = DMCRCR(M,M,LI2,M,M,M,HH44,M,M,M,F,M)
    HH44 = DMCRCR(M,M,LI,M,M,M,F,M,M,M,HH44,M)

    for i in range(M10):
        for j in range(M20):
            GG3[i,j]=0.0

    for i in range(M):
        for j in range(M):
            GG3[i,j]=-HH11[i,j]+HH21[i,j]
            GG3[i,j+2*M]=-HH13[i,j]+HH23[i,j]
            GG3[i,j+3*M]=-Z[i,j]
            GG3[i+M,j+M]=-HH32[i,j]+HH42[i,j]
            GG3[i+M,j+2*M]=-Y[i,j]
            GG3[i+M,j+3*M]=-HH34[i,j]+HH44[i,j]

    K1=0
    K0=0

    for j in range(M20):

        if IH[j] == 1:
            K1=K1+1
        if IH[j] == 0:
            if IH[j] == 0:
                K0=K0+1
            if IH[j] == 1:
                continue

            for i in range(M10):
                GG5[i,K0]=GG3[i,j]

        else:
            for i in range(M10):
                GG4[i,K1]=-1*GG3[i,j] # Original was -GG3[i,j]
                B10[K1]=B5[j]


    B6 = DMUCRV (M10,M10,GG4,M10,M10,B10,1,M10,B6)
    B7 = DLSLCG(M10,GG5,M10,B6,1,B7)
    K1=0

    for j in range(M20):
        if IH[j] == 0:
            K1=K1+1
        if IH[j] == 1:
            continue

        B5[j]=B7[K1]

    for i in range(M):
        UK1[i]=B5[i]
        AIK1[i]=B5[i+M]

    AA = DMUCRV(M,M,Z,M,M,AIK1,1,M,AA)
    BB = DMUCRV(M,M,Y,M,M,UK1,1,M,BB)
    M = DMCRCR(M,M,LI,M,M,M,LI3,M,M,M,CC,M)
    M = DMCRCR(M,M,LI,M,M,M,LI2,M,M,M,DD,M)

    for i in range(M):
        B1[i]=UK1[i]
        B1[i+M]=AIK1[i]
        B1[i+2*M]=AA[i]
        B1[i+3*M]=BB[i]

    B4 = DLSLCG(M20,GG,M20,B1,1,B4)

    for i in range(M):
        AA[i]=0.
        B[i]=0.
        B[i]=B4[i]

    AA = DMUCRV(M,M,LU3,M,M,B,1,M,AA)

    for i in range(M):
        BB[i]=0.
        B[i]=0.
        B[i]=B4[i+M]

    BB = DMUCRV(M,M,LU2,M,M,B,1,M,BB)

    for i in range(M):
        UX[i]=AA[i]+BB[i]
        if LM == MMT:
            UK1[i]=UX[i]
        UXM[i]=DSQRT(REAL(UX[i])**2+AIMAG(UX[i])**2)

    for i in range(M):
        AA[i]=0.
        B[i]=0.
        B[i]=B4[i+2*M]

    AA = DMUCRV(M,M,LI3,M,M,B,1,M,AA)

    for i in range(M):
        BB[i]=0.
        B[i]=0.
        B[i]=B4[i+3*M]

    BB = DMUCRV(M,M,LI2,M,M,B,1,M,BB)

    for i in range(M):

        AIX[i]=AA[i]+BB[i]
        if LM == MMT:
            AIK1[i]=AIX[i]
        AIXM[i]=np.sqrt(AIX[i].real**2+AIX[i].imag**2)

        if I == 1 and LL == 1 and PR == 2:
            PPP1[NN,LL]=PPP1[NN,LL]+AIXM[0]**2/2*R11[0]
        if I == 1 and LL > 1:
            PPP1[NN,LL]=PPP1[NN,LL]+AIXM[0]**2/2*R11[0]
        if I == 2 and LL == 1 and PR == 2:
            PPP2[NN,LL]=PPP2[NN,LL]+AIXM[1]**2/2*R11[1]
        if I == 2 and LL > 1:
            PPP2[NN,LL]=PPP2[NN,LL]+AIXM[1]**2/2*R11[1]
        if I == 3 and LL == 1 and PR == 2:
            PPP3[NN,LL]=PPP3[NN,LL]+AIXM[2]**2/2*R11[2]
        if I == 3 and LL > 1:
            PPP3[NN,LL]=PPP3[NN,LL]+AIXM[2]**2/2*R11[2]
        if I == 4 and LL == 1 and PR == 2:
            PPP4[NN,LL]=PPP4[NN,LL]+AIXM[3]**2/2*R11[3]
        if I == 4 and LL > 1:
            PPP4[NN,LL]=PPP4[NN,LL]+AIXM[3]**2/2*R11[3]
        if I == 5 and LL == 1 and PR == 2:
            PPP5[NN,LL]=PPP5[NN,LL]+AIXM[4]**2/2*R11[4]
        if I == 5 and LL > 1:
            PPP5[NN,LL]=PPP5[NN,LL]+AIXM[4]**2/2*R11[4]
        if I == 6 and LL == 1 and PR == 2:
            PPP6[NN,LL]=PPP6[NN,LL]+AIXM[5]**2/2*R11[5]
        if I == 6 and LL > 1:
            PPP6[NN,LL]=PPP6[NN,LL]+AIXM[5]**2/2*R11[5]
        if I == 7 and LL == 1 and PR == 2:
            PPP7[NN,LL]=PPP7[NN,LL]+AIXM[6]**2/2*R11[6]
        if I == 7 and LL > 1:
            PPP7[NN,LL]=PPP7[NN,LL]+AIXM[6]**2/2*R11[6]
        if I == 8 and LL == 1 and PR == 2:
            PPP8[NN,LL]=PPP8[NN,LL]+AIXM[7]**2/2*R11[7]
        if I == 8 and LL > 1:
            PPP8[NN,LL]=PPP8[NN,LL]+AIXM[7]**2/2*R11[7]
        if LL == 1 and PR == 2.:
            PPP[NN,LL]=PPP[NN,LL]+AIXM[I]**2/2*R11[I]
        if LL > 1:
            PPP[NN,LL]=PPP[NN,LL]+AIXM[I]**2/2*R11[I]
        if LL == 1 and PR == 1.:
            PP1=PP1+AIXM[I]**2/2*R11[I]
        if LL == 1 and PR == 2.:
            PP2=PP2+AIX[I]**2/2*R11[I]
        SM[i]=UX[i]*np.conj(AIX[i])/2.

    # Writing UXM and AIXM values into 10th and 11th modules

# Implementing other FORTRAN's subroutines
def DLINRG(N, A, LDA, AINV, LDAINV):
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


def DMRRRR(NRA, NCA, A, LDA, NRB, NCB, B, LDB, NRC, NCC, C, LDC):
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


def DMCRCR(NRA, NCA, A, LDA, NRB, NCB, B, LDB, NRC, NCC, C, LDC):
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


def DEVLCG(N, A, LDA, EVAL):
    """
    Compute all of the eigenvalues of a complex matrix.

        N — Order of the matrix A.   (Input)
        A — Complex matrix of order N.   (Input)
        LDA — Leading dimension of A exactly as specified in the dimension statement in the calling program.   (Input)
        EVAL —  Complex vector of length N containing the eigenvalues of A in decreasing order of magnitude.   (Output)
    """
    return LA.eigvals(A)


def DLFTCG(N, A, LDA, FAC, LDFAC, IPVT):
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
    return lu_factor(A)


def LFSCG(N, FAC, LDFAC, IPVT, B, IPATH, X):
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


def DLFDCG(A, N, FAC, LDFAC, IPVT, DET1, DET2):
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


def DLINCG(N, A, LDA, AINV, LDAINV):
    """ !!! SCALE FIRST TO 1/N !!!
    Compute the inverse of a complex general matrix.

        N — Order of the matrix A.   (Input)
        A — Complex N by N matrix containing the matrix to be inverted.   (Input)
        LDA — Leading dimension of A exactly as specified in the dimension statement of the calling program.   (Input)
        AINV — Complex N by N matrix containing the inverse of A.   (Output)
        If A is not needed, A and AINV can share the same storage locations.
        LDAINV — Leading dimension of AINV exactly as specified in the dimension statement of the calling program.   (Input)
    """
    if N:
        A = A*1/N
    return np.linalg.inv(A)


def DMUCRV(NRA, NCA, A, LDA, NX, X, IPATH, NY, Y):
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


def DLSLCG(N, A, LDA, B, IPATH, X):
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
    return np.linalg.inv(A).dot(X)


# Au = np.matmul(Z, Y)
# Ai = np.matmul(Y, Z)
