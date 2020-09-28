import argparse
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from loguru import logger

class SVMScratch():
    def __init__(self, C, kernel, degree, coef0, epsilon, nepoch):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.coef0 =coef0
        self.epsilon = epsilon
        self.nepoch = nepoch

    def fit(self, X, y):
        """模型训练"""
        self.init_params(X, y)

        self.smo_outer(y)

        self.sv_idx = np.squeeze(np.argwhere(self.alpha > 0))
        self.sv = X[self.sv_idx]
        self.sv_y = y[self.sv_idx]
        self.sv_alpha = self.alpha[self.sv_idx]

    def predict(self, x):
        """模型预测"""
        n_sv = self.sv.shape[0]
        x_kernel = np.zeros(n_sv)
        for i in range(n_sv):
            x_kernel[i] = self.kernel_transform(self.sv[i], x)
        y_pred = np.dot(self.sv_alpha * self.sv_y, x_kernel)
        return 1 if y_pred > 0 else -1

    def smo_outer(self, y):
        """smo外层循环"""
        num_epoch = 0
        traverse_trainset = True
        alpha_change = 0
        while alpha_change > 0 or traverse_trainset:
            alpha_change = 0
            if traverse_trainset:
                for i in range(self.m):
                    alpha_change += self.smo_inner(i, y)
            else:
                bound_sv_idx = np.nonzero(np.logical_and(self.alpha > 0, self.alpha < self.C))[0]
                for i in bound_sv_idx:
                    alpha_change += self.smo_inner(i, y)
            num_epoch += 1
            if num_epoch >= self.nepoch:
                break
            if traverse_trainset:
                traverse_trainset = False
            elif alpha_change == 0:
                #没有找到合适的alpha
                traverse_trainset = True

    def smo_inner(self, i, y):
        """smo内层循环"""
        if (self.violate_kkt(i,y)):
            Ei = self.calc_E(i,y)
            j, Ej = self.select_j(i,y)

            alpha_i_old = self.alpha[i].copy()
            alpha_j_old = self.alpha[j].copy()

            if y[i] != y[j]:
                L = max(0, self.alpha[j] - self.alpha[i])
                H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
            else:
                L = max(0, self.alpha[j] + self.alpha[i] - self.C)
                H = min(self.C, self.alpha[j] + self.alpha[i])
            if L == H:
                return 0

            eta = self.K[i,i] + self.K[j,j] - 2 * self.K[i,j]
            if eta <= 0:
                return 0

            self.alpha[j] += y[j] *(Ei -Ej) / eta
            self.alpha[j] = np.clip(self.alpha[j], L, H)
            self.update_E(j,y)

            if abs(self.alpha[j] - alpha_j_old) < 0.00001:
                return 0

            self.alpha[i] += y[i] * y[j] * (alpha_j_old -self.alpha[j])
            self.update_E(i,y)

            b1_new = self.b - Ei - y[i]*self.K[i,i]*(self.alpha[i]-alpha_i_old) - y[j]*self.K[i,j]*(self.alpha[j]-alpha_j_old)
            b2_new = self.b - Ej -y[i]*self.K[i,j]*(self.alpha[i]-alpha_i_old) - y[j]*self.K[j,j]*(self.alpha[j] - alpha_j_old)

            if self.alpha[i] > 0 and self.alpha[i] < self.C:
                self.b = b1_new
            elif self.alpha[j] > 0 and self.alpha[j] < self.C:
                self.b = b2_new
            else:
                self.b = (b1_new + b2_new) / 2
            return 1
        else:
            return 0

    def select_j(self, i, y):
        """选择第二个变量"""
        Ei = self.calc_E(i,y)
        self.ecache[i] = [1, Ei]

        max_diff = -np.inf
        max_j = -1
        max_Ej = -np.inf
        ecache_idx = np.nonzero(self.ecache[:,0])[0]

        if len(ecache_idx) > 1:
            for j in ecache_idx:
                if j == i:
                    continue
                Ej = self.calc_E(i,y)
                diff = abs(Ei - Ej)
                if diff > max_diff:
                    max_diff = diff
                    max_j = j
                    max_Ej = Ej
            return max_j, max_Ej
        else:
            j = i
            while j == i:
                j = random.randint(0, self.m-1)
            Ej = self.calc_E(j,y)
            return j, Ej


    def violate_kkt(self, i, y):
        """是否违反KKT条件"""
        if self.alpha[i] > 0 and self.alpha[i] < self.C:
            return abs(y[i] * self.g(i,y) - 1) < self.epsilon
        return  True

    def g(self, i, y):
        return np.dot(self.alpha * y, self.K[i]) + self.b

    def calc_E(self, i, y):
        return  self.g(i,y) - y[i]
    def update_E(self, i, y):
        Ei = self.calc_E(i,y)
        self.ecache[i] = [1, Ei]


    def init_params(self, X, y):
        """初始化参数"""
        self.m = X.shape[0]
        self.n = X.shape[1]
        self.alpha = np.zeros(self.m)
        self.b = 0

        self.ecache = np.zeros((self.m, 2))
        self.K = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                self.K[i,j] = self.kernel_transform(X[i], X[j])

    def kernel_transform(self, x1, x2):
        gamma = 1 / self.n
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'ploy':
            return np.power(gamma * np.dot(x1, x2) + self.coef0, self.degree)
        else:
            return np.exp(-gamma * np.dot(x1-x2, x1-x2))

def main():
    parser = argparse.ArgumentParser(description='感知机算法Scratch')
    parser.add_argument('--nepoch', type=int, default=3000, help='训练多少个epoch后终止训练')
    parser.add_argument('--C', type=float, default=1, help='正则化项系数')
    parser.add_argument('--kernel', type=str, default='rbf', help='核函数类型')
    parser.add_argument('--degree', type=int, default=3, help='多项式核函数次数')
    parser.add_argument('--coef0', type=float, default=0, help='多项式核函数中系数')
    parser.add_argument('--epsilon', type=float, default=0.001, help='检验第一个变量对应的样本是否违反KKT条件的检验范围')
    args = parser.parse_args()

    X, y = load_iris(return_X_y=True)
    y[:50] = -1
    xtrain, xtest, ytrain, ytest = train_test_split(X[:100], y[:100], train_size=0.8, shuffle=True)

    model = SVMScratch(args.C, args.kernel, args.degree, args.coef0, args.epsilon, args.nepoch)
    model.fit(xtrain, ytrain)

    n_test = xtest.shape[0]
    n_right = 0
    for i in range(n_test):
        y_pred = model.predict(xtest[i])
        if y_pred == ytest[i]:
            n_right += 1
        else:
            logger.info('该样本真是标签为：{}， 但是Scratch模型预测标签为： {}'.format(ytest[i], y_pred))
    logger.info('Scratch模型在测试集上的准确率为：{}%'.format(n_right * 100 / n_test))

    skmodel = SVC(args.C, args.kernel, args.degree, coef0=args.coef0, max_iter=args.nepoch)
    skmodel.fit(xtrain, ytrain)
    logger.info('sklearn模型在测试集上的准确率为：{}%'.format(100 * skmodel.score(xtest, ytest)))

if __name__ == '__main__':
    main()
    """还是直接用sklearn的模型吧！"""


