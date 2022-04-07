import numpy as np
from numpy import heaviside, linspace, sqrt
from scipy.stats import kurtosis, skew, expon, gamma, chi2
import matplotlib.pyplot as plt
from statsmodels.distributions import ECDF
from scipy.optimize import minimize


def parsing_file(filename: str) -> np.array:
    res = np.array([])
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            row = line.split()
            row_len = len(row)
            row_array = np.zeros(row_len)
            for i in range(row_len):
                row_array[i] = float(row[i])
            if len(res) > 0:
                res = np.concatenate((res, row_array))
            else:
                res = row_array

        file.close()
    return res


def gamma_minimize(cdf2check, int_num: int, a: float, b: float, nums):
    borders = np.linspace(min_value, max_value, int_num + 1)
    N = len(nums)
    res = 0
    for i in range(int_num):
        p_k = cdf2check(borders[i + 1], a=a, scale=b) - cdf2check(borders[i], a=a, scale=b)
        v_k = len([num for num in nums if borders[i] < num < borders[i + 1]])
        res += (v_k - N * p_k) ** 2 / (N * p_k)

    return res


def exponential_minimize(cdf2check, int_num: int, a: float, nums):
    borders = np.linspace(min_value, max_value, int_num + 1)
    N = len(nums)
    res = 0
    for i in range(int_num):
        p_k = cdf2check(borders[i + 1], scale=a) - cdf2check(borders[i], scale=a)
        v_k = len([num for num in nums if borders[i] < num < borders[i + 1]])
        res += (v_k - N * p_k) ** 2 / (N * p_k)

    return res


def tcdf(x: float, X: np.array) -> float:
    return 1. / len(X) * sum([heaviside(x - X_i, 1) for X_i in X])


file_name = "Number_12.txt"
data = parsing_file(file_name)

n = len(data)
sample_mean = np.mean(data)
sample_var = np.var(data)
sample_std = np.std(data)
asymmetry_coefficient = skew(data)  # через функцию встроенную в scipy
kurtosis_coefficient = kurtosis(data)

# подсчет выборочных характеристик
print("# sample mean: " + str(sample_mean))
print("# sample variance: " + str(sample_var))
print("# sample std : " + str(sample_std))
print("# asymmetry coefficient: " + str(asymmetry_coefficient))
print("# kurtosis coefficient: " + str(kurtosis_coefficient))

# нормированная гистограмма и э. ф. р.
# обычная гистограмма без каких-либо преобразований
fig, ax = plt.subplots()
plt.hist(data)
plt.title("Базовая гистограмма выборки, без каких-либо прообразований")
plt.grid()

# уже нормальная, нормированная гистограмма
delta = 0.01  # delta добавляется для того, чтобы интервалы были строгими
max_value, min_value = max(data) + delta, min(data) - delta
n_intervals = 10
normalized_data = data
bins = list(np.linspace(min_value, max_value, n_intervals))

fig, ax = plt.subplots()
plt.hist(normalized_data, density=True, bins=bins)
plt.title("Нормированная гистограмма")
plt.grid()

# другой способ построения гистограммы

ecdf = ECDF(data)
fig, ax = plt.subplots()
plt.plot(ecdf.x, ecdf.y, lw=2, label="empirical density function")
plt.title("Эмпирическая функция распределения")
plt.grid()
plt.legend()

# довертельные полосы
gamma_1, gamma_2 = 0.9, 0.95  # доверительные вероятности
# пока примем квантиль 1-0.90=0.05 равным 1.22, а квантиль 1-0.95 равным 1.36
x = linspace(0, 7, 200)
u_gamma = 1.22
# polosa_090_low = [tcdf(x_i, normalized_data) - u_gamma / sqrt(len(normalized_data)) for x_i in x]
# polosa_090_up = [tcdf(x_i, normalized_data) + u_gamma / sqrt(len(normalized_data)) for x_i in x]

line_090_low = ecdf.y - u_gamma / sqrt(len(normalized_data))
line_090_up = ecdf.y + u_gamma / sqrt(len(normalized_data))

fig, ax = plt.subplots()
plt.plot(ecdf.x, line_090_low, lw=2, label="lower border", c="r")
plt.plot(ecdf.x, ecdf.y, lw=2, label="empirical density function", c="b")
plt.plot(ecdf.x, line_090_up, lw=2, label="upper border", c="k")
plt.title("0.9-доверительная полоса")
plt.grid()
plt.legend()

u_gamma = 1.36
# polosa_090_low = [tcdf(x_i, normalized_data) - u_gamma / sqrt(len(normalized_data)) for x_i in x]
# polosa_090_up = [tcdf(x_i, normalized_data) + u_gamma / sqrt(len(normalized_data)) for x_i in x]

line_095_low = ecdf.y - max(u_gamma / sqrt(len(normalized_data)), 0)
line_095_up = ecdf.y + min(u_gamma / sqrt(len(normalized_data)), 1)

fig, ax = plt.subplots()
plt.plot(ecdf.x, line_095_low, lw=2, label="lower border", c="r")
plt.plot(ecdf.x, ecdf.y, lw=2, label="empirical density function", c="b")
plt.plot(ecdf.x, line_095_up, lw=2, label="upper border", c="k")
plt.title("0.95-доверительная полоса")
plt.grid()
plt.legend()

# Сравнение полос
fig, ax = plt.subplots()
plt.plot(ecdf.x, line_095_low, lw=2, label="border-0.95", c="r")
plt.plot(ecdf.x, ecdf.y, lw=2, label="empirical density function", c="k")
plt.plot(ecdf.x, line_095_up, lw=2, c="r")
plt.plot(ecdf.x, line_090_low, lw=2, label="border-0.9", c="b")
plt.plot(ecdf.x, line_090_up, lw=2, c="b")
plt.title("Сравнение доверительных полос")
plt.grid()
plt.legend()
# plt.show()

# ну а теперь осталось проверить вид распределения,  предполагаем, что это либо гамма, либо экспоненчицальное
intervals_quants = np.array([5, 10, 20])
expon_estim, gamma_estim = [], []
for number_intervals in intervals_quants:
    alpha = 0.02  # уровень значимости
    chii2 = lambda x: gamma_minimize(gamma.cdf, number_intervals, a=x[0], b=x[1], nums=data)
    result = minimize(chii2, np.array([1.6, 0.9]), method='BFGS', tol=1e-15)
    print(f"Value: {result['fun']}, min: {result['x']}")
    gamma_estim.append(result['x'])
    print(result['fun'] < chi2.ppf(1 - alpha, number_intervals - 3))
    print("# size = " + str(number_intervals))
    print("chi^2 quantile = " + str(chi2.ppf(1 - alpha, number_intervals - 3)))
    print("t-statistics = " + str(result['fun']))

print("##### expon")
for number_intervals in intervals_quants:
    alpha = 0.05
    chii2 = lambda x: exponential_minimize(expon.cdf, number_intervals, a=x, nums=data)
    result = minimize(chii2, np.array([0.3]), method='TNC', tol=1e-15)
    print(f"Value: {result['fun']}, min: {result['x']}")
    expon_estim.append(result['x'])
    print(result['fun'] < chi2.ppf(1 - alpha, number_intervals - 2))
    print("# size = " + str(number_intervals))
    print("chi^2 quantile = " + str(chi2.ppf(1 - alpha, number_intervals - 2)))
    print("t-statistics = " + str(result['fun']))

# считаем эмперические оценки параметров распределений

lambda_ = np.mean(np.array(expon_estim), axis=0)
a, b = np.mean(np.array(gamma_estim), axis=0)
print(lambda_)
print(str(a) + "   " + str(b))

# сравниваем гистограммы

fig, ax = plt.subplots()
x_ = linspace(min_value, max_value, 100)
plt.hist(normalized_data, density=True, bins=bins, label="normal hist")
plt.plot(x_, expon.pdf(x_, scale=lambda_), label="exponential distribution", c="r")
plt.plot(x_, gamma.pdf(x_, a=a, scale=b), label="gamma distribution", c="k")
plt.title("Сравнение гипотетических теоретических кривых распределений")
plt.grid()
plt.legend()
# plt.show()

# сравниваем функции распределений

fig, ax = plt.subplots()
x_ = linspace(min_value, max_value, 100)
plt.plot(ecdf.x, ecdf.y , label="normal hist")
plt.plot(x_, expon.cdf(x_, scale=lambda_), label="exponential distribution", c="r")
plt.plot(x_, gamma.cdf(x_, a=a, scale=b), label="gamma distribution", c="k")
plt.title("Сравнение теоретических и эмперической плотностей распределения")
plt.grid()
plt.legend()
plt.show()
