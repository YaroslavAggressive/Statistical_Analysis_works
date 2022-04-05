import numpy as np
from numpy import sqrt, fabs
from numpy.linalg import inv, matrix_rank
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.stats import norm, f, t


def parsing_file(filename: str) -> np.array:
    with open(filename, 'r') as file:
        lines = file.readlines()
        rows = [line.split() for line in lines]
        n_, m_ = len(lines), len(rows[0])
        table = np.zeros((n_, m_))
        for i in range(n_):
            for j in range(m_):
                table[i, j] = float(rows[i][j])
        file.close()
    return table


y_filename = "Y_15.txt"
X_filename = "X.txt"

X = parsing_file(X_filename)
y = parsing_file(y_filename)
# print(y)
# print(X)

# делаю вариант № 12

# дальше ищем мнк-оценки
# первая оценка - для коэффициентов
rank = matrix_rank(X)  # смотрим, что он равен 4, и все окей
print("X-matrix rank is: " + str(rank))
X_T = np.transpose(X)
X_mul = np.matmul(X_T, X)
X_1 = inv(X_mul)
a = np.matmul(X_1, X_T).dot(y)
print("# a")
print(a)

# теперь ищем оценку дисперсии

y_estim = X.dot(a)
print("# y_estim")
print(y_estim)
s_2 = np.dot(np.transpose(y - y_estim), (y - y_estim)) / (X.shape[0] - X.shape[1])
print("# s^2")
print(s_2)

# далее матрицу ковариации
cov_a = s_2 * X_1
print("# cov(a)")
print(cov_a)  # С КРЫШЕЧКОЙ!!!!

# здесь будем считать оценки параметров s_i

print("# s(a_i)")
X_sqrt = sqrtm(X_1)
s_estim = np.zeros(len(a))
for i in range(len(a)):
    s_i = sqrt(s_2) * X_sqrt[i, i]
    print("s_" + str(i) + ": " + str(s_i))
    s_estim[i] = s_i
print("# s(a_i): ")
print(s_estim)

# здесь построим матрицу корреляции
X_1_shape = X_1.shape
print(X_1_shape)
corr_a = np.zeros(X_1_shape)

for i in range(X_1_shape[0]):
    for j in range(X_1_shape[1]):
        corr_a[i, j] = X_1[i, j] / sqrt(X_1[i, i] * X_1[j, j])

print("# corr_a")
print(corr_a)

# построим гистограмму отстатков
idxs = np.arange(len(y))
err = y - y_estim

x_axis = np.arange(-7, 7, 0.01)
fig, ax = plt.subplots()
plt.plot(x_axis, norm.pdf(x_axis, loc=0, scale=1), 'k-', lw=2, label='frozen pdf')
plt.hist(err, density=True, histtype='stepfilled', alpha=0.8, label="residuals histogram")
ax.legend(loc='best', frameon=False)
plt.xlabel("t")
plt.ylabel("|y - y_estim|")
plt.grid()
plt.show()

# далее считаем коэффициент детерминации (смещенный и несмещенный)

y_mean = np.mean(y)
print("# y mean")
print(y_mean)

e = y - y_estim
print("# error")
print(e)

n, m = X.shape

R_2 = 1 - np.sum(np.square(e)) / np.sum(np.square(y - y_mean))
print("# R^2")
print(R_2)
R_H_2 = 1 - (np.sum(np.square(e)) / (n - m)) / (np.sum(np.square(y - y_mean)) / (n - 1))
print("# R_H^2")
print(R_H_2)

# графики реальной и оцененной регрессии:

fig, ax = plt.subplots()
indices = np.arange(len(X))
plt.plot(indices, y, c="b", label="reference value")
plt.plot(indices, y_estim, c="k", label="regression estimate")
for ind, y_i in zip(indices, y):
    ax.scatter(ind, y_i, c="b")
for ind, y_estim_i in zip(indices, y_estim):
    ax.scatter(ind, y_estim_i, c="k")
plt.xlabel("Number of experiment")
plt.ylabel("Experiment output value")
plt.legend()
plt.grid()
plt.show()

# индивидуальные доверительные интервалы

tau = 0.9
alpha = 0.45
for i in range(len(a)):
    print(str(i) + " параметр регрессии, индивидуальный доверительный интервал: " +
          str([a[i] - s_estim[i] * t.ppf(1 - alpha / 2, n - m), a[i] + s_estim[i] * t.ppf(1 - alpha / 2, n - m)]))

# проверим равенство нулю всех оценок коэффициентов

for i, s_i in enumerate(s_estim):
    alpha = 0.45  # уровень значимости
    t_quant = t.ppf(1 - alpha / 2, n - m)
    t_ = fabs(-a[i]) / s_i
    print("Проверка гипотезы, что " + str(i + 1) + " параметр регрессии равен 0: ")
    print("# t_" + str(i) + " = " + str(t_))
    print("# quantile" + " = " + str(t_quant))
    if t_ > t_quant:
        print("Гипотезу Н0 мы отвергаем, так как статистика превосходит квантиль распределния Стьюдента")
    else:
        print("Гипотезу Н0 принимаем, так как статистика не превосходит квантиля распределния Стьюдента")


# найдем совместную доверительную область на основании неравенства Чебышева

alpha = 0.45
tau = 0.23
for i in range(len(a)):
    print(str(i) + " параметр регрессии, доверительный интервал: " + str([a[i] - tau * s_estim[i],
                                                                          a[i] + tau * s_estim[i]]))

# проверим на адекватность модель среднего

ssssss = 0

# проверка гипотезы об идентичности двух регрессий
np.random.seed(1)
n_1, n_2 = 8, 7
range_ind = np.arange(len(X))
# ind_x1 = [2 * i for i in range(len(X) // 2 + 1)]
# ind_x2 = [2 * i + 1 for i in range(len(X) // 2)]
ind_x1 = sorted(np.random.choice(range_ind, n_1))
ind_x2 = [i for i in range_ind if i not in ind_x1]
X1 = X[ind_x1, :]
X2 = X[ind_x2, :]
y1 = y[ind_x1, :]
y2 = y[ind_x2, :]

a1 = np.matmul(inv(np.matmul(np.transpose(X1), X1)), np.transpose(X1)).dot(y1)
print("# a_1:")
print(a1)
a2 = np.matmul(inv(np.matmul(np.transpose(X2), X2)), np.transpose(X2)).dot(y2)
print("# a_2:")
print(a2)
a_R = np.matmul(inv(np.matmul(np.transpose(X), X)), np.transpose(X)).dot(y)
print("# a_R:")
print(a_R)

Q_1 = np.dot(np.transpose(y1 - X1.dot(a1)), (y1 - X1.dot(a1)))
Q_2 = np.dot(np.transpose(y2 - X2.dot(a2)), (y2 - X2.dot(a2)))
Q_R = np.dot(np.transpose(y - X.dot(a_R)), (y - X.dot(a_R)))

s2 = (Q_1 + Q_2) / (n_1 + n_2 - 2 * m)
print("# Q_R: ")
print(Q_R)
print("# Q_1: ")
print(Q_1)
print("# Q_2: ")
print(Q_2)
print("# s^2: ")
print(s2)

alpha = 0.45  # уровень значимости
tt = (Q_R - Q_1 - Q_2) / m / s2
print("# t-statistics")
print(tt)

dfn, dfd = m, n_1 + n_2 - 2 * m
t_alpha = f.ppf(1 - alpha, dfn, dfd)
print("# quantile")
print(t_alpha)

print("# Проверка гипотезы на идентичность двух регрессий")
if tt > t_alpha:
    print("Гипотезу Н0 мы отвергаем, так как статистика превосходит квантиль распределния Фишера")
else:
    print("Гипотезу Н0 принимаем, так как статистика не превосходит квантиля распределния Фишера")

# fig, ax = plt.subplots()
# plt.plot(ind_x1, X1.dot(a1), c="b", label="output estimation")
# plt.plot(ind_x1, y1, c="k", label="output values")
# plt.title("Регрессионная модель на выборке X1")
# for ind, y_i in zip(ind_x1, y1):
#     ax.scatter(ind, y_i, c="b")
# for ind, y_estim_i in zip(ind_x1, X1.dot(a1)):
#     ax.scatter(ind, y_estim_i, c="k")
# plt.legend()
# plt.grid()
# plt.xlabel("t")
# plt.ylabel("Y(t)")
#
# fig, ax = plt.subplots()
# plt.title("Регрессионная модель на выборке X2")
# plt.plot(ind_x2, X2.dot(a2), c="b", label="output estimation")
# plt.plot(ind_x2, y2, c="k", label="output values")
# for ind, y_i in zip(ind_x2, y2):
#     ax.scatter(ind, y_i, c="b")
# for ind, y_estim_i in zip(ind_x2, X2.dot(a2)):
#     ax.scatter(ind, y_estim_i, c="k")
# plt.legend()
# plt.grid()
# plt.xlabel("t")
# plt.ylabel("Y(t)")
# plt.show()

# дальше будем прогнозировать
j = np.random.choice(range_ind, 1)
x_j, y_j = X[j], y[j]
prognoz_ind = [i for i in range(n) if i != j]
X_prog = X[prognoz_ind]
y_prog = y[prognoz_ind]
a_14 = np.matmul(inv(np.matmul(np.transpose(X_prog), X_prog)), np.transpose(X_prog)).dot(y_prog)
y_estim_j = x_j @ a_14
y_estim__ = X.dot(a_14)

# оценка прогноза
print("# a_14 = " + str(a_14))
print("# оценка прогноза ")
print(y_estim_j)
print("# ошибка ее вычисления")
print(fabs(y_estim_j - y_j))
print("# доля ошибки")
print(fabs(y_estim_j - y_j) / y_j)
print("# интервальная оценка прогноза")
# оценки полученной модели
print("# оценки полученной модели")
s_2_prog = np.dot(np.transpose(y_prog - X_prog.dot(a_14)), (y_prog - X_prog.dot(a_14))) / (n - 1 - m)
print("# s^2: " + str(s_2_prog))
alpha = 0.45
s_a = np.zeros(len(a_14))
for i in range(len(a_14)):
    s_ii = sqrt(s_2_prog) * sqrtm(inv(np.matmul(np.transpose(X_prog), X_prog)))[i, i]
    s_a[i] = s_ii
print("# оценки отклонений параметров регрессии")
print(s_a)
cov_a = s_2_prog * inv(np.matmul(np.transpose(X_prog), X_prog))
print("# cov(a)")
print(cov_a)  # С КРЫШЕЧКОЙ!!!!

t_alpha = t.ppf(1 - alpha / 2, n - 1 - m)
s_tau_2 = s_2_prog * (np.dot(x_j, inv(np.matmul(np.transpose(X_prog), X_prog)).dot(np.transpose(x_j))) + 1)
print("D_i = [" + str(y_estim_j - sqrt(s_tau_2) * t_alpha) + ", " + str(y_estim_j + sqrt(s_tau_2) * t_alpha) + " ]")

# сравнение графиков исходной регрессии и уменьшеной регрессии

# fig, ax = plt.subplots()
# plt.title("Сравнение регрессионных моделей: исходной и использованной для прогнозирования")
#
# plt.plot(indices, y, c="k", label="true output")
# plt.plot(indices, y_estim, c="b", label="y estimation by 15 elements")
# plt.plot(indices, y_estim__, c="r", label="y estimation by 14 elements")
# plt.grid()
# plt.legend()
# plt.ylabel("X * a")
# plt.xlabel("t")
# plt.show()
