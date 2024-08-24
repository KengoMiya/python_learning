import numpy as np

class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state =1):
        self.eta = eta #学習率
        self.n_iter = n_iter #訓練回数
        self.random_state = random_state #シャッフル用の乱数生成器
    
    def fit(self, X, y):
        rgen = np.random.Randomstate(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0) #誤差を追加
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        #総入力を計算
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
