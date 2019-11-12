<center>哈尔滨工业大学计算机科学与技术学院</center>
<center style="font-size:30px">实验报告 </center><br></br>
<center>课程名称： 机器学习</center>
<center>课程类型： 选修</center>
<center>实验题目：  k-means 聚类方法和GMM混合高斯模型</center><br/>
<center>学号：1171000405</center>
<center>姓名：许天骁</center>
<div STYLE="page-break-after: always;"></div>

#### 一、实验目的
&ensp;&ensp;实现一个 K-means 算法和混合高斯模型，并且用 EM 算法估计模型中的参数。
#### 二、实验要求及实验环境

##### 实验要求
1. 用高斯分布产生 k 个高斯分布的数据（不同均值和方差，其中参数自己设定）
2. 用 K-means 聚类，测试效果
3. 用混合高斯模型和你实现的 EM 算法估计参数，看看每次迭代后似然值变化情况，
4. 考察 EM 算法是否可以获得正确的结果（与你设定的结果比较）

##### 实验环境
1. 硬件环境：X64CPU; 4.0Ghz; 16G RAM
2. 软件环境：Windows 64位
3. 开发环境：Python 3.6.8(64位); Anaconda3; Visual Stuio Code

#### 三、设计思想（本程序中的用到的主要算法及数据结构）
##### 程序结构
使用Python进行编写，总共分为两个程序块，生成数据（ $makedata.py$ ）和程序的主部分（ $main.py$ ）
##### 问题描述
已知 $ N $个包含 $ m $ 个特征的数据分布在代数空间内。对于任意一个数据点，需要将其划分在一个等价类之中。等价类的数目可手动改变。
##### 算法介绍
###### k-means
概述：k均值聚类算法（k-means clustering algorithm）是一种迭代求解的聚类分析算法。该算法可以分为两步。首先随机确定 $k$ 个中心点。每一轮迭代第一步：对于每一个数据点，找到距离（此处采用欧氏距离） $k$ 个中心点最近的一个。将其划分在其等价类集合中。第二步：对于所有等价类集合，重新计算该集合的中心点。实现中心点的更新。迭代到一定的条件下便停止迭代。终止条件可以是：没有数据点被更改等价类集合或者聚类的中心点不再发生变化。

###### GMM和EM算法
###### GMM
高斯混合模型（Gaussian Mixed Model）指的是多个高斯分布函数的线性组合，理论上GMM可以拟合出任意类型的分布，通常用于解决同一集合下的数据包含多个不同的分布的情况。
对于一个多维的随机变量，其混合高斯模型可以有如下表示
$$
P(\boldsymbol{x}) = \sum_{k=1}^K\pi_kN(\boldsymbol{x}|\boldsymbol{\mu}_k, \Sigma_k)
$$
其中 $N(\boldsymbol{x}|\mu_k, \Sigma_k)$ 是混合高斯模型中的一个分量。表达式如下：
$$
N(\boldsymbol{x}|\mu_k, \Sigma_k)=\frac{1}{\sqrt{(2\pi)^2|\Sigma|}}\exp(-\frac{1}{2}(\boldsymbol{x-\mu}^T)\Sigma^{-1}(\boldsymbol{x-\mu}))
$$
表示其中 $\pi_k$ 是混合系数，可以理解成取到第 $k$ 个分量的概率。 $\mu_k$ 是第 $k$ 个分量的均值， $\Sigma_k$ 第 $k$ 个分量的协方差矩阵。其中 $\pi_k$ 满足
$$
\sum_{k=1}^K\pi_k = 1,0<\pi_k<1
$$
###### EM
EM算法，是一种通过迭代的方式求解GMM中各个参数的方法。该算法的思想是首先假设确定各个高斯分布分量的占比 $\pi_k$ ，接着根据极大似然估计的方法，计算出每一个高斯分布分量的的参数 $\mu_k, \Sigma_k$.接着根据新的参数重新估计 $\pi_k$ 完成迭代过程。该迭代方法是可以被证明是可以收敛的。接下来给出大致计算过程。
首先提出隐变量 $\boldsymbol{z}$ 该变量是一个one-hot型的向量，第k个维度若为1则表示当前的变量被当前模型分配到了第k组。此时 $\pi_k$ 可以理解为先验概率，即 $P(\boldsymbol{z})$/同时可以得到条件概率：
$$
P(\boldsymbol{x}|\boldsymbol{z}) = \prod_{k=1}^K N(\boldsymbol{x}|\boldsymbol{\mu}_k, \Sigma_k)^\boldsymbol{z_k}
$$
此时获得了先验概率 $P(\boldsymbol{z})=\pi_k$ 条件概率 $P(\boldsymbol{x}|\boldsymbol{z})$ 和联合概率 $P(\boldsymbol{x})$ 就可以得到后验概率 $P(z_i=k|\boldsymbol{x}_i,\boldsymbol{\mu}_k, \Sigma_k)$（这一步在EM算法中被称为E步）：
$$
\gamma(z_{n,k}) = \frac{\pi_kN(\boldsymbol{x}|\boldsymbol{\mu}_k, \Sigma_k)}{\sum_{j=1}^N\pi_jN(\boldsymbol{x}|\boldsymbol{\mu}_j, \Sigma_j)}
$$
表示在 $\boldsymbol{x}$ 取固定值时，第 $n$ 个$\boldsymbol{z}$ 的第 $k$ 个维度取值为1时的概率，直观上可以理解成，第 $n$ 个点被分配至第 $k$ 个类别的概率。有了这个 $\gamma$ 矩阵之后，便可以对 $\mu 和 \Sigma$ 进行极大似然估计了。
对全概率公式进行改写一下，可以得到多个参数共同影响的全概率密度公式
$$
P(\boldsymbol{x}|\pi, \mu, \Sigma) = \sum_{k=1}^K\pi_kN(\boldsymbol{x}|\mu_k, \Sigma_k)
$$
对于整体样本的极大似然估计，只需要将每一个样本的概率进行乘积运算，同时得到的就是似然函数。
$$
L(X|\Theta)=\sum_{i}\ln\sum_{k}\pi_kN(\boldsymbol{x}|\boldsymbol{\mu}_k, \Sigma_k)
$$
利用往常的MLE算法，直接对需要优化的参数求导即可，但是，在该似然函数中，对数运算中包含加法运算，所以可以先对似然函数进行下界的优化，利用Jenson不等式：
$$
f[E(x)] \geq E[f(x)]
$$
代入公式，进行下界优化
$$
\begin{aligned}
    L(X|\Theta)&=\sum_{i}\ln\sum_{k}\pi_kN(\boldsymbol{x}_i|\boldsymbol{\mu}_k, \Sigma_k)\\
    &=\sum_{i}\ln\sum_{k}P(z_i=k|\boldsymbol{x}_i,\boldsymbol{\mu}_k, \Sigma_k)\frac{\pi_kN(\boldsymbol{x}_i|\mu_k, \Sigma_k)}{P(z_i=k|\boldsymbol{x}_i,\boldsymbol{\mu}_k, \Sigma_k)}\\
    &\geq\sum_{i}\sum_{k}P(z_i=k|\boldsymbol{x}_i,\boldsymbol{\mu}_k, \Sigma_k)\ln\frac{\pi_kN(\boldsymbol{x}_i|\mu_k, \Sigma_k)}{P(z_i=k|\boldsymbol{x}_i,\boldsymbol{\mu}_k, \Sigma_k)}\\
    &=\sum_{i}\sum_{k}\gamma_{i,k}\ln\frac{\pi_kN(\boldsymbol{x}_i|\boldsymbol{\mu_k}, \Sigma_k)}{\gamma_{i,k}}
\end{aligned}
$$
由此可以开始进行迭代，不断增大该函数的上界，间接使得似然函数最大化。
1. 参数初始化
   由于EM算法是属于初值敏感的算法，所以使用K-means得到的结果进行初始化三个参数（ $\boldsymbol{\mu}_k, \boldsymbol{\pi}, \Sigma_k$ ）
2. E步
   根据当前的 $\boldsymbol{\pi},  \boldsymbol{\mu}_k, \Sigma_k$ 计算出新的 $\Gamma$ 矩阵
$$
\gamma_{i,k} = \frac{\pi_kN(\boldsymbol{x}_i|\mu_k, \Sigma_k)}{\sum_{j=1}^N\pi_jN(\boldsymbol{x}_i|\mu_j, \Sigma_j)}
$$
3. M步
   对上述的似然函数进行优化，由于这是一个限制条件下的优化问题，所以可以使用拉格朗日算子法得到等价优化问题：
   $$
   L(X|\Theta) = \sum_{i}\sum_{k}\gamma_{i,k}\ln\frac{\pi_kN(\boldsymbol{x}_i|\boldsymbol{\mu_k}, \Sigma_k)}{\gamma_{i,k}}+\lambda(\sum_{k}\pi_k-1)
   $$
   分别对三个参数求导，得到迭代公式：
   $$
    \boldsymbol\mu_k=\frac{\sum_i\gamma_{i,k}\boldsymbol x_i}{\sum_i\gamma_{i,k}}
   $$
   $$
    \pi_k=\frac{\sum_i\gamma_{i,k}}{N}
   $$
   $$
   \Sigma_k=\frac{\sum_i\gamma_{i,k}(\boldsymbol x_i-\boldsymbol\mu_k)^T(\boldsymbol x_i-\boldsymbol\mu_k)}{\sum_i\gamma_{i,k}}
   $$
#### 四、实验结果与分析
 - 数据生成
  根据输入的参数可以分别生成协方差矩阵为对角阵，或者根据输入的协方差矩阵生成椭圆形的数据。
  协方差矩阵不是对角阵时，四组点的均值分别为：
  $ (1,1) (1,4) (4,1) (4,4) $
 - 变量及解释
   $num_x$ : 每一组生成的点的个数
   $epoch$ : 训练的轮数
   $kind$ : 实际分类组数
   $k$ : 聚类时分类的组数
   $Sigma$ : 协方差矩阵
 - 实验结果
  K-means：
  K-means算法是初值敏感的算法，所以随着程序的不同次运行，会分别出现分类良好和不良好的情况，如下图所示：
  <img src="./mingan.png">
  <img src="./不敏感.png">
  其中紫色点是又K-means给出的各类点的均值点。由于随机性，可以看到第一张图分类不良好，第二张图完成了良好的聚类效果。
  GMM和EM：
  由于利用了K-means的计算结果，所以GMM可以更好做到聚类效果。下面给出实验结果。
  首先是在K-means已经完成了良好分类的情况下的似然函数的初始值和迭代结束的值:
  <img src="./e.png">

  |epoch | $\mu$ | 似然函数值|
  | :-: | :-: | :-: |
  | 0    | -  | $2.680$ |
  |19|$(0.968, 4.01) (4.03, 1.02)(1.01, 0.996)(3.95, 3.98)$|$2.615$
  接着是在K-means发生了初值敏感的情况下的似然函数的初始值和迭代结束的值:
  <img src="./e1.png">
  |epoch | $\mu$ | 似然函数值|
  | :-: | :-: | :-: |
  | 0    | -  | $3.039$ |
  |19|$(1.01, 0.992) (4.05, 0.976)(0.960,3.99)(4.16,3.94)$|$2.600$
 - 分析
   1. k-means及GMM均可较好地完成简单的聚类问题。
   2. k-means初值敏感，容易陷入局部最优解
   3. GMM利用k-means初始化参数后，可以很好的优化k-means计算出来的结果。证明GMM可以跳过局部最优解，寻找到全局最优解。

#### 五、附录：源代码

```python
#生成数据部分( makedata.py )
import numpy as np
import matplotlib.pyplot as plt

def get_data(num_x, num_k, x_mean, y_mean, sigma):
    num = int(num_k * num_x)
    X = np.zeros(num)
    Y = np.zeros(num)
    for i in range(num_k):
        for j in range(num_x):
            X[i * num_x + j] = np.random.normal(x_mean[i], sigma[i])
            Y[i * num_x + j] = np.random.normal(y_mean[i], sigma[i])
    X = X[:, np.newaxis]
    Y = Y[:, np.newaxis]
    return X, Y
def get_blob(num_x, num_k, miu, sigma):
    S = np.mat(sigma[0])
    R = np.linalg.cholesky(S)
    x = np.dot(np.random.randn(num_x, 2), R) + np.ones((num_x, 1)).dot(miu[0])
    for i in range(num_k - 1):
        S = np.mat(sigma[i + 1])
        R = np.linalg.cholesky(S)
        x = np.append(x, np.dot(np.random.randn(num_x, 2), R) + np.ones((num_x, 1)).dot(miu[i + 1]), axis=0)
    return np.array(x[:, 0]), np.array(x[:, 1])
```

```python
#主函数部分( main.py )
import numpy as np
import makedata as data
import matplotlib.pyplot as plt

args = dict(
    x_mean=[0.2, 0.5, 0.6, 0.3, 0.7],
    y_mean=[0.2, 0.5, 0.2, 0.4, 0.5],
    sigma=[0.055, 0.055, 0.075, 0.065, 0.085],
    miu=[[[1, 1]],
         [[4, 4]],
         [[1, 4]],
         [[4, 1]]],
    Sigma=[[[0.2, -0.1],
            [-0.1, 0.2]],
           [[0.7, -0.4],
            [-0.4, 0.5]],
           [[0.2, 0.1],
            [0.1, 0.2]],
           [[0.3, -0.2],
           [-0.2, 0.2]]],
    C=['ro', 'go', 'bo', 'yo', 'ko'],
    epoch=20,
    k=4,
    num_x=100,
    kind=4,
)
epoch = args['epoch']
num_x = args['num_x']
kind = args['kind']
k = args['k']
x_mean = args['x_mean']
y_mean = args['y_mean']
sigma = args['sigma']
C = args['C']
Sigma = args['Sigma']
miu = args['miu']

X, Y = data.get_data(num_x, kind, x_mean, y_mean, sigma)
X, Y = data.get_blob(num_x, kind, miu, Sigma)
x = np.vstack((X.T, Y.T)).T
num = int(num_x * kind)

def dis(x1, y1, x2, y2):
    return np.square(x1 - x2) + np.square(y1 - y2)

def k_means(view):# num * 1
    if view:
        plt.ion()
    k_x = np.ones((num, 1)).dot(np.random.random((1, k)))#num * k
    k_y = np.ones((num, 1)).dot(np.random.random((1, k)))#num * k
    for i in range(epoch):
        if view:
            plt.cla()
        dis = pow(X - k_x, 2) + pow(Y- k_y, 2)
        group = dis.argmin(axis = 1)
        x_mean = np.zeros((k, 1))
        y_mean = np.zeros((k, 1))
        for i in range(num):
            x_mean[group[i]] += X[i]
            y_mean[group[i]] += Y[i]
        for i in range(k):
            if sum(group == i) == 0:
                x_mean[i] = np.random.rand()
                y_mean[i] = np.random.rand()
            else:
                x_mean[i] /= sum(group == i)
                y_mean[i] /= sum(group == i)
        k_x = np.ones((num, 1)).dot(x_mean.T)
        k_y = np.ones((num, 1)).dot(y_mean.T)
        if view:
            for i in range(k):
                plt.plot(x_mean[i], y_mean[i], 'mo')
            for i in range(num):
                plt.plot(X[i], Y[i], C[group[i]])
            plt.pause(0.3)    
    return (group, x_mean, y_mean)

def get_pi_miu_sig(g, x_mean, y_mean):
    mean = np.append(x_mean, y_mean, axis=1)
    pi = np.zeros(k)
    sig = np.zeros((k, 2, 2))
    for i in range(num):
        pi[group[i]] += 1
    for i in range(k):
        sig[i] = np.cov(X[i * num_x : (i + 1) * num_x].T, Y[i * num_x : (i + 1) * num_x].T)
    return pi/num, mean, sig

def pr_F(x_i, miu_j, sig_j):
    sig_det = np.linalg.det(sig_j)
    x_i = np.mat(x_i)
    miu_j = np.mat(miu_j)
    first = (((2 * np.pi) ** 2) * sig_det) ** 0.5
    point = np.exp((-0.5) * (x_i - miu_j).dot(np.linalg.inv(sig_j)).dot((x_i - miu_j).T))
    return (point / first).A[0]

def E_setp(pi, mean, sig):
    gama = np.zeros((num, k))
    for i in range(num):
        base = 0
        for j in range(k):
            base += pi[j] * pr_F(x[i], mean[j], sig[j])
        for j in range(k):
            gama[i][j] = pi[j] * pr_F(x[i], mean[j], sig[j]) / base
    return gama

def M_step(gama):
    resig = np.zeros((k, 2, 2))
    N = np.ones((2, num)).dot(gama)
    remiu = gama.T.dot(x) / N.T
    repi = np.ones((1, num)).dot(gama)[0] / num
    for i in range(k):
        for j in range(num):
            resig[i] += gama[j][i] * (x[j] - remiu[i])[:, np.newaxis].dot((x[j] - remiu[i])[np.newaxis, :])
        resig[i] /= np.ones((1, num)).dot(gama)[0][i]
    return repi, remiu, resig

group, x_mean, y_mean = k_means(view=True)
pi, mean, sig = get_pi_miu_sig(group, x_mean, y_mean)

def likelyhood(pi, mean, sig):
    re = 0
    sig_det = np.linalg.det(sig)
    for i in range(num):
        tmp = 0
        for j in range(k):
            tmp += pi[j] * pr_F(x[i], mean[j], sig[j])
        re += np.log(tmp)
    return -re/num

for i in range(epoch):
    print("epoch=",i)
    gam = E_setp(pi, mean, sig)
    pi, mean, sig = M_step(gam)
    print(likelyhood(pi, mean, sig))
print("pi =",pi)
print("mean =",mean)
print("sigma =",sig)

plt.ioff()
plt.show()
```