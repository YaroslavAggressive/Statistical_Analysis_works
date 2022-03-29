import numpy as np
from numpy.linalg import solve
from numpy.random import multivariate_normal
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def rolling_test_method(sample_1: np.array, sample_2: np.array, dim: int) -> list:
    predicted = []
    for i, elem in enumerate(sample_1):
        sample_copy = np.delete(sample_1, i, axis=0)
        params = count_empiric_params(dim, sample_copy, sample_2)
        pred_tmp = discriminant_classification(np.array([elem]), params[0], params[1])
        predicted.append(pred_tmp)
    for i, elem in enumerate(sample_2):
        sample_copy = np.delete(sample_2, i, axis=0)
        params = count_empiric_params(dim, sample_1, sample_copy)
        pred_tmp = discriminant_classification(np.array([elem]), params[0], params[1])
        predicted.append(pred_tmp)
    return predicted


def count_discriminant_coeffs(mean_first: np.array, mean_second: np.array, cov: np.array) -> (np.array, float):
    b = mean_first - mean_second
    alpha = solve(cov, b)
    return alpha, sum(alpha * mean_first + alpha * mean_second) / 2


def discriminant_classification(sample: np.array, alpha: np.array, const: float) -> list:
    y_pred = []
    for i in range(len(sample)):
        y_pred.append(0 if sum(alpha * sample[i]) >= const else 1)
    return y_pred


def count_empiric_params(dim: int, distrib_1: np.array, distrib_2: np.array) -> list:
    mean_first = np.mean(distrib_1, axis=0)
    mean_second = np.mean(distrib_2, axis=0)

    cov_1 = np.cov(distrib_1, rowvar=False)
    cov_2 = np.cov(distrib_2, rowvar=False)

    size_1, size_2 = len(distrib_1), len(distrib_2)
    cov = 1. / (size_2 + size_1 - 2) * (cov_1 * (size_1 - 1) + cov_2 * (size_2 - 1))
    alpha, c = count_discriminant_coeffs(mean_first, mean_second, cov)

    xi_1 = sum(np.mean(alpha * distrib_1, axis=0))
    xi_2 = sum(np.mean(alpha * distrib_2, axis=0))

    sigma_z = 0
    for i in range(dim):
        for j in range(dim):
            sigma_z += alpha[i] * alpha[j] * cov[i, j]

    return [alpha, c, xi_1, xi_2, sigma_z]


# def create_model_data(mean_first: np.array, mean_second: np.array, cov: np.array, test_size: int,
#                       train_size: int, name: str) -> list:
#
#     # Доделать метод, чтобы выглядело по-человечески
#
#     first_train_sample = multivariate_normal(mean_first, cov, train_size)
#     second_train_sample = multivariate_normal(mean_second, cov, train_size)
#
#     first_test_sample = multivariate_normal(mean_first, cov, test_size)
#     second_test_sample = multivariate_normal(mean_second, cov, test_size)
#
#     # draw samples
#     # train sample by class
#     fig_tmp = plt.figure()
#     ax_tmp = fig_tmp.add_subplot(111, projection='3d')
#     ax_tmp.set_title("Easily separable model data, train sample")
#     for point in x_multi:
#         ax_tmp.scatter(*point, c="r")
#
#     for point in y_multi:
#         ax_tmp.scatter(*point, c="b")
#     # plt.show()
#
#     # test sample by class
#     fig_tmp = plt.figure()
#     ax_tmp = fig_tmp.add_subplot(111, projection='3d')
#     ax_tmp.set_title("Easily separable model data, test sample")
#     for point in test_class_1:
#         ax_tmp.scatter(*point, c="r")
#
#     for point in test_class_2:
#         ax_tmp.scatter(*point, c="b")
#     # plt.show()
#
#     return [first_train_sample, second_train_sample, first_test_sample, second_test_sample]


# # first part lab data parameters
# np.random.seed(20)
# p = 3
# mean_1 = np.array([1, 2, 3])
# mean_2 = np.array([3, 4, 5])
# size_test = 100
# size_train = 200
#
# cov_X = np.array([[1, -0.1, 0.2], [-0.1, 3, 0.3], [0.2, 0.3, 4]])
#
# print("good variant")
# x_multi = multivariate_normal(mean_1, cov_X, size_train)
# y_multi = multivariate_normal(mean_2, cov_X, size_train)
#
# test_class_1 = multivariate_normal(mean_1, cov_X, size_test)
# test_class_2 = multivariate_normal(mean_2, cov_X, size_test)
#
# # draw samples
# # train sample by class
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title("Easily separable model data, train sample")
# for coord in x_multi:
#     ax.scatter(*coord, c="r")
#
# for coord in y_multi:
#     ax.scatter(*coord, c="b")
# # plt.show()
#
# # test sample by class
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title("Easily separable model data, test sample")
# for coord in test_class_1:
#     ax.scatter(*coord, c="r")
#
# for coord in test_class_2:
#     ax.scatter(*coord, c="b")
# # plt.show()
#
# true = [0 for i in range(size_test)] + [1 for j in range(size_test)]
#
# # classification
#
# parameters = count_empiric_params(p, x_multi, y_multi)
# print("Alpha = " + str(parameters[0]))
# print("C = " + str(parameters[1]))
# print("Ksi_1 = " + str(parameters[2]))
# print("Ksi_2 = " + str(parameters[3]))
#
# pred = discriminant_classification(test_class_1, parameters[0], parameters[1]) +\
#        discriminant_classification(test_class_2, parameters[0], parameters[1])
#
# table = confusion_matrix(true, pred)
# display = ConfusionMatrixDisplay(table, display_labels=["D1", "D2"])
# display.plot()
# display.ax_.set_title("Classification of test sample")
# # plt.show()
#
# # next estimate the error probabilities in three ways
# # simply by the percentage of misclassified
# print("Оценки ошибок в хорошем случае классификации тестов")
# p_2_1 = table[1, 0] / sum(table[1])
# p_1_2 = table[0, 1] / sum(table[0])
#
# print(p_1_2)
# print(p_2_1)
#
# # use function for train samples
#
# pred = discriminant_classification(x_multi, parameters[0], parameters[1]) +\
#        discriminant_classification(y_multi, parameters[0], parameters[1])
#
# true = [0 for _ in range(size_train)] + [1 for _ in range(size_train)]
# table = confusion_matrix(true, pred)
# display = ConfusionMatrixDisplay(table, display_labels=["D1", "D2"])
# display.plot()
# display.ax_.set_title("Classification of training sample")
# # plt.show()
#
# # scores
#
# delta_square = (parameters[2] - parameters[3])**2 / parameters[4]
# D_H = float(2 * size_train - p - 3) / (2 * size_train - 2) * delta_square - p * 2 / size_train
# print("delta^2 = " + str(delta_square))
# print("D_H = " + str(D_H))
#
# # next estimate the error probabilities in three ways
# # simply by the percentage of misclassified
# print("Оценки ошибок в хорошем случае классификации тренировок")
# p_2_1 = table[1, 0] / sum(table[1])
# p_1_2 = table[0, 1] / sum(table[0])
#
# print(p_1_2)
# print(p_2_1)
#
# # laplace function - counted by hand
#
# # contining probabilities by rolling exam method
#
# pred = rolling_test_method(x_multi, y_multi, p)
# table = confusion_matrix(true, pred)
# p_2_1 = table[1, 0] / sum(table[1])
# p_1_2 = table[0, 1] / sum(table[0])
#
# print("# Вероятности ошибки классификации по скользящему экзамену:")
# print(p_1_2)
# print(p_2_1)
#
# # now bad separated data
#
# print("bad variant")
# mean_1 = np.array([1, 2, 3])
# mean_2 = np.array([1.2, 2.3, 3.4])
#
# x_multi = multivariate_normal(mean_1, cov_X, size_train)
# y_multi = multivariate_normal(mean_2, cov_X, size_train)
#
# test_class_1 = multivariate_normal(mean_1, cov_X, size_test)
# test_class_2 = multivariate_normal(mean_2, cov_X, size_test)
#
# # draw model distribution
# # train sample by class
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title("Badly separable model data, train sample")
# for coord in x_multi:
#     ax.scatter(*coord, c="r")
#
# for coord in y_multi:
#     ax.scatter(*coord, c="b")
# # plt.show()
#
# # test sample by class
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title("Badly separable model data, test sample")
# for coord in test_class_1:
#     ax.scatter(*coord, c="r")
#
# for coord in test_class_2:
#     ax.scatter(*coord, c="b")
# # plt.show()
#
# true = [0 for _ in range(size_test)] + [1 for _ in range(size_test)]
#
# # classification after parameters evaluation
#
# parameters = count_empiric_params(p, x_multi, y_multi)
# print("Alpha = " + str(parameters[0]))
# print("C = " + str(parameters[1]))
# print("Ksi_1 = " + str(parameters[2]))
# print("Ksi_2 = " + str(parameters[3]))
#
# pred = discriminant_classification(test_class_1, parameters[0], parameters[1]) +\
#        discriminant_classification(test_class_2, parameters[0], parameters[1])
#
# table = confusion_matrix(true, pred)
# display = ConfusionMatrixDisplay(table, display_labels=["D1", "D2"])
# display.plot()
# display.ax_.set_title("Classification of test sample")
# # plt.show()
#
# # next estimate the error probabilities in three ways
# # simply by the percentage of misclassified
# print("Оценки ошибок в плохом случае классификации тестов")
# p_2_1 = table[1, 0] / sum(table[1])
# p_1_2 = table[0, 1] / sum(table[0])
#
# print(p_1_2)
# print(p_2_1)
#
# # classification on train sample
#
# pred = discriminant_classification(x_multi, parameters[0], parameters[1]) +\
#        discriminant_classification(y_multi, parameters[0], parameters[1])
#
# true = [0 for _ in range(size_train)] + [1 for _ in range(size_train)]
# table = confusion_matrix(true, pred)
# display = ConfusionMatrixDisplay(table, display_labels=["D1", "D2"])
# display.plot()
# display.ax_.set_title("Classification of training sample")
# # plt.show()
#
# # scores
#
# delta_square = (parameters[2] - parameters[3])**2 / parameters[4]
# D_H = float(2 * size_train - p - 3) / (2 * size_train - 2) * delta_square - p * 2 / size_train
# print("delta^2 = " + str(delta_square))
# print("D_H = " + str(D_H))
#
# # next estimate the error probabilities in three ways
# # simply by the percentage of misclassified
# print("Оценки ошибок в плохом случае классификации тренировок")
# p_2_1 = table[1, 0] / sum(table[1])
# p_1_2 = table[0, 1] / sum(table[0])
#
# print("# Относительное число неверно классифицированных объектов:")
# print(p_1_2)
# print(p_2_1)
#
# # laplace function - count by hnd and table
#
# # counting probability by rolling exam method
#
# pred = rolling_test_method(x_multi, y_multi, p)
# table = confusion_matrix(true, pred)
# p_2_1 = table[1, 0] / sum(table[1])
# p_1_2 = table[0, 1] / sum(table[0])
# print("# Вероятности ошибки классификации по скользящему экзамену:")
# print(p_1_2)
# print(p_2_1)
