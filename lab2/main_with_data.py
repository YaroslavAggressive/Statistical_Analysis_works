import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from numpy.linalg import eig

from main import count_discriminant_coeffs


def rolling_test_method(sample_1: np.array, sample_2: np.array, dim: int) -> list:
    predicted = []
    for i, elem in enumerate(sample_1):
        sample_copy = np.delete(sample_1, i, axis=0)
        params = count_empiric_params(dim, sample_copy, sample_2)
        pred_tmp = discriminant_classification(np.array([elem]), params[3], params[4])
        predicted.append(pred_tmp)
    for i, elem in enumerate(sample_2):
        sample_copy = np.delete(sample_2, i, axis=0)
        params = count_empiric_params(dim, sample_1, sample_copy)
        pred_tmp = discriminant_classification(np.array([elem]), params[3], params[4])
        predicted.append(pred_tmp)
    return predicted


def pca(data: np.array) -> np.array:
    mean = np.mean(data, axis=0)
    data_meaned = data - mean
    cov = np.cov(data_meaned, rowvar=False)

    eig_nums, eig_vectors = eig(cov)
    for i, (num, vect) in enumerate(zip(eig_nums, eig_vectors)):  # debug output for research
        print(str(i) + " eigenvalue: " + str(num))
    # видно, что среди дисперсий компонент выделяются три параметра,
    # остальные в сравнении с ними уже будут слабо влиять, поэтому будем понижать число компонент до 3
    dim_new = 3
    sorted_index = np.argsort(eig_nums)[::-1]
    sorted_eigenvectors = eig_vectors[:, sorted_index]
    eig_vectors_subset = sorted_eigenvectors[:, 0:dim_new]
    data_reduced = np.dot(eig_vectors_subset.T, data_meaned.T).T
    return data_reduced


def discriminant_classification(sample: np.array, alpha: np.array, const: float) -> list:
    y_pred = []
    for i in range(len(sample)):
        y_pred.append(1 if sum(alpha * sample[i]) >= const else 2)
    return y_pred


def count_empiric_params(dim: int, distrib_1: np.array, distrib_2: np.array) -> list:
    mean_1 = np.mean(distrib_1, axis=0)
    mean_2 = np.mean(distrib_2, axis=0)

    cov_1 = np.cov(distrib_1, rowvar=False)
    cov_2 = np.cov(distrib_2, rowvar=False)

    size_1, size_2 = len(distrib_1), len(distrib_2)
    cov = 1. / (size_1 + size_2 - 2) * ((size_1 - 1) * cov_1 + (size_2 - 1) * cov_2)
    alpha, c = count_discriminant_coeffs(mean_1, mean_2, cov)

    xi_1 = sum(np.mean(alpha * distrib_1, axis=0))
    xi_2 = sum(np.mean(alpha * distrib_2, axis=0))

    sigma_z = 0
    for j in range(dim):
        for k in range(dim):
            sigma_z += alpha[j] * alpha[k] * cov[j, k]

    return [mean_1, mean_2, cov, alpha, c, xi_1, xi_2, sigma_z]


def research_repo_data(data: np.array, with_pca=False):
    # dataset class separation
    if with_pca:
        print("Using PCA for german-data: ")
        reduced_data = pca(data[:, :-1])
        class_separation = data[:, -1]
        data = np.insert(reduced_data, len(reduced_data[0]), class_separation, axis=1)
    class_1 = np.array(data[data[:, -1] == 1])[:, :-1]  # последним слайсом отрезаем уже зармеченный признак класса
    class_2 = np.array(data[data[:, -1] == 2])[:, :-1]  # чтобы уже с ним не возиться потом потом
    test_size_1 = 200  # test sample size, other - go for training classifier
    test_size_2 = 100
    train_class_1 = class_1[test_size_1:]
    train_class_2 = class_2[test_size_2:]
    test_data = np.concatenate((class_1[: test_size_1], class_2[: test_size_2]), axis=0)
    p = data.shape[1] - 1

    # testing on test data (3.1)
    true = [1 for _ in range(test_size_1)] + [2 for _ in range(test_size_2)]
    params = count_empiric_params(p, train_class_1, train_class_2)
    print("Mean first class: " + str(params[0]))
    print("Mean second class: " + str(params[1]))
    print("Cov: " + str(params[2]))
    print("Z: " + str(params[3]))
    print("C: " + str(params[4]))
    print("Xi_1: " + str(params[5]))
    print("Xi_2: " + str(params[6]))
    print("Sigma^2: " + str(params[7]))
    pred = discriminant_classification(test_data, params[3], params[4])

    # build statistics

    table = confusion_matrix(true, pred)
    display = ConfusionMatrixDisplay(table, display_labels=["D1", "D2"])
    display.plot()
    display.ax_.set_title("Classification of test sample of german data")
    plt.show()

    print("Оценки ошибок классификации тестовой выборки")
    p_2_1 = table[1, 0] / sum(table[1])
    p_1_2 = table[0, 1] / sum(table[0])

    print(p_1_2)
    print(p_2_1)

    # use function for train samples (3.2)

    pred = discriminant_classification(train_class_1, params[3], params[4]) + discriminant_classification(train_class_2,
                                                                                                          params[3],
                                                                                                          params[4])
    true = [1 for _ in range(len(train_class_1))] + [2 for _ in range(len(train_class_2))]
    table = confusion_matrix(true, pred)
    display = ConfusionMatrixDisplay(table, display_labels=["D1", "D2"])
    display.plot()
    display.ax_.set_title("Classification of training sample of german data")
    plt.show()

    if with_pca:
        colors = {1: "r", 2: "b"}
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Classification of test data")
        for point, cls in zip(test_data, pred):
            ax.scatter(*point, c=colors[cls])
        plt.show()

    # scores (2.2)

    delta_square = (params[5] - params[6]) ** 2 / params[7]
    D_H = (test_size_1 + test_size_2 - p - 3) / (test_size_1 + test_size_2 - 2) * delta_square - p * \
          (1. / test_size_1 + 1. / test_size_2)
    print("delta^2 = " + str(delta_square))
    print("D_H = " + str(D_H))

    # draw model distribution
    # train sample by class
    print("# Теперь оцениваем тренировочные данные: ")
    p_2_1 = table[1, 0] / sum(table[1])
    p_1_2 = table[0, 1] / sum(table[0])
    print("# Относительное число неверно классифицированных объектов:")
    print(p_1_2)
    print(p_2_1)

    # laplace function - count by hnd and table

    # counting probability by rolling exam method

    pred = rolling_test_method(train_class_1, train_class_2, p)
    table = confusion_matrix(true, pred)
    p_2_1 = table[1, 0] / sum(table[1])
    p_1_2 = table[0, 1] / sum(table[0])
    print("# Вероятности ошибки классификации по скользящему экзамену:")
    print(p_1_2)
    print(p_2_1)


# интеграл лапласа
np.random.seed(20)
filename = "german.data-numeric"
data = np.loadtxt(filename)
research_repo_data(data)
# здесь потестим pca
research_repo_data(data, with_pca=True)
