<center>哈尔滨工业大学计算机科学与技术学院</center>
<center style="font-size:30px">实验报告 </center><br></br>
<center>课程名称：机器学习</center>
<center>课程类型：选修</center>
<center>实验题目：PCA模型实验</center><br/>
<center>学号：1171000405</center>
<center>姓名：许天骁</center>
<div STYLE="page-break-after: always;"></div>

#### 一、实验目的
&ensp;&ensp;实现一个PCA模型，能够对给定数据进行降维（即找到其中的主成分）
#### 二、实验要求及实验环境
##### 实验要求
1. 首先人工生成一些数据（如三维数据），让它们主要分布在低维空间中，如首先让某个维度的方差远小于其它唯独，然后对这些数据旋转。生成这些数据后，用你的PCA方法进行主成分提取。
2. 找一个人脸数据（小点样本量），用你实现PCA方法对该数据降维，找出一些主成分，然后用这些主成分对每一副人脸图像进行重建，比较一些它们与原图像有多大差别（用信噪比衡量）。

##### 实验环境
1. 硬件环境：X64CPU; 4.0Ghz; 16G RAM
2. 软件环境：Windows 64位
3. 开发环境：Python 3.6.8(64位); Anaconda3; Visual Stuio Code

#### 三、设计思想（本程序中的用到的主要算法及数据结构）
##### 程序结构
使用Python进行编写，总共分为两个程序块，生成数据（ $makedata.py$ ）和程序的主部分（ $main.py$ ）
##### 问题描述
PCA（Principal Component Analysis）主成分分析。
在 $n$ 维向量空间内，存在 $m$ 个变量 $X$ 现在需要找到 $k$ 组线性无关的基向量，做这些变量在这些基向量的投影。用于提取数据的主要特征，常用于高维数据的降维。
##### 算法介绍
对于一组变量
$$
X=
 \left[
 \begin{matrix}
   x_1^{(1)} & x_1^{(2)} & \cdots & x_1^{(n)} \\
   x_2^{(1)} & x_2^{(2)} & \cdots & x_2^{(n)} \\
   x_3^{(1)} & x_3^{(2)} & \cdots & x_3^{(n)} \\
   \vdots & \vdots & \ddots & \vdots &\\
   x_m^{(1)} & x_n^{(2)} & \cdots &x_m^{(n)}
  \end{matrix}
  \right]
$$
其协方差矩阵表示了两个维度之间的方差，由于需要找到 $k$ 组线性无关的基向量，所以可以得知需要对原 $X$ 的协方差矩阵进行对角化。而特征值分解就可以做到这一部分。可以证明特征值就是对角化后的矩阵的对角线上的值。因此选取前 $k$ 大的特征值对应的特征向量作为基向量即可对原数据进行降维。
#### 四、实验结果与分析
##### 数据生成
  根据输入的参数可以生成一组三维的点，是一个添加了一些随机数的旋转过后的立体的对数螺线。大致观测如下：
  <img src="./1.png" width = 50%>
  <img src="./2.png" width = 50%>
  <img src="./3.png" width = 50%>
  除了生成的数据，同样使用了mnist手写数据集作为真实数据进行测试。
  使用了手写数字0进行降维，图片的大小是 28 x 28 的灰度图像，选取了5000张进行试验。下面展示的是第一张图片。
  <img src="./5.jpg">
 - 变量及解释
   $num\_point$ : 生成的点的个数
   $miu$ : 噪声的均值
   $sigma$ : 噪声的方差

##### 实验结果
 - 生成数据
  生成的数据进行降维后的结果（三维数据将为二维数据）
  <img src="./4.png" width = 50%>
 - 真实数据
  下面给出了信噪比随着降低的维度变化的曲线。
  <img src="./信噪比.png">
  下面是降低到1维再恢复后的图片和信噪比(PSNR = 12.93)
  <img src="./6.png">
  下面是降低到100维再恢复后的图片和信噪比(PSNR = 24.52)
  <img src="./7.png">
  下面是降低到350维再恢复后的图片和信噪比(PSNR = 39.23)
  <img src="./8.png">
  下面是降低到500维再恢复后的图片和信噪比(PSNR = 74.53)
  <img src="./9.png">
  下面是降低到600维再恢复后的图片和信噪比(PSNR = 259.9)
  <img src="./10.png">

##### 分析
 - 生成的数据
  可以看到降维后的结果比较接近生成的对数螺线的形状。比较好完成了降维工作。
 - 真实的数据
  信噪比的定义：
  PSNR高于40dB说明图像质量极好（即非常接近原始图像），
  在30—40dB通常表示图像质量是好的（即失真可以察觉但可以接受），
  在20—30dB说明图像质量差；
  最后，PSNR低于20dB图像不可接受
  观察上述的实验结果，降低到非常低的维度（1维）再恢复至原图的效果基本与图像的均值相同。但是信噪比非常低（12.93）表示与原图的差距十分巨大。降低至100维再恢复得到的图像，虽然可以比较清晰地看出形状，但是锐度不够，即图像的分界线和背景没有原图明显。降低到350维左右再恢复图像即可达到40dB左右的信噪比。根据实验结果可以看到基本与原图相同了。而继续的500维和600维的实验可以看出信噪比超过40dB以后，已经几乎无法看出原图和压缩后的图片的区别了。所以对于mnist数据集进行350维的压缩，基本可以提取出图片的主成分，完成降维。
#### 五、附录：源代码

```python
#生成数据部分( makedata.py )
import numpy as np

def makedata(n, miu, sigma, theta):
    a = 1
    b = 0.2
    th=np.linspace(0, 10, int(n / 10))
    x = np.expand_dims(np.tile(a * np.exp(b * th) * np.cos(th), 10) + np.random.normal(miu, sigma, (n)), axis=0)
    y = np.expand_dims(np.tile(a * np.exp(b * th) * np.sin(th), 10) + np.random.normal(miu, sigma, (n)), axis=0)
    z = np.expand_dims(np.array(np.linspace(-1.5, 1.5, n) + np.random.normal(miu, sigma, (n))), axis=0)
    mo = np.concatenate([np.concatenate([np.concatenate([x, y], axis=0), z], axis=0), np.ones((1, n))], axis=0)
    ro_x = np.zeros((4, 4))
    ro_x[0][0] = 1.0
    ro_x[1][1] = np.cos(theta)
    ro_x[1][2] = -np.sin(theta)
    ro_x[2][1] = np.sin(theta)
    ro_x[2][2] = np.cos(theta)
    ro_x[3][3] = 1.0
    ro_y = np.zeros((4, 4))
    ro_y[0][0] = np.cos(theta)
    ro_y[1][1] = 1
    ro_y[0][2] = -np.sin(theta)
    ro_y[2][0] = np.sin(theta)
    ro_y[2][2] = np.cos(theta)
    ro_y[3][3] = 1.0
    mo = np.dot(ro_x, mo)
    mo = np.dot(ro_y, mo)
    return np.delete(mo, -1, axis=0).T
```

```python
#主函数部分( main.py )
import numpy as np
import makedata as data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

args = dict(
    num_point=200,
    miu=0,
    sigma=0.3,
    ro_theta=0.5235987755982988730771,
    mode='mnist',
    num_x=5000,
    feature=600
)#pi/6
num_point = args['num_point']
miu = args['miu']
sigma = args['sigma']
ro_theta = args['ro_theta']
mode = args['mode']
num_x = args['num_x']
feature = args['feature']

def point():
    pic = data.makedata(num_point, miu, sigma, ro_theta)
    mean = np.mean(pic, 0)
    pic -= mean
    cov = pic.T.dot(pic) / num_point
    lamda, v = np.linalg.eig(cov)
    index = lamda.argsort()[-2:][::-1]
    sub_v = v.T[index]
    pic_k = sub_v.dot(pic.T)

    fig1 = plt.figure()
    ax = fig1.gca(projection='3d')
    ax.scatter(pic.T[0], pic.T[1], pic.T[2])
    fig2 = plt.figure()
    plt.scatter(pic_k[0], pic_k[1])
    plt.show()


def psnr(target, ref, scale):
    target_data = np.array(target)
    ref_data = np.array(ref)
    diff = ref_data - target_data
    rmse = np.sqrt(np.mean(diff ** 2.) )
    return 20*np.log10(255.0/rmse)

def mnist():
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original', data_home='./PCA/data/')
    raw_pic = np.float64(mnist['data'][:num_x])
    path = './PCA/image/'
    mean = np.mean(raw_pic, 0)
    #tmpim = Image.fromarray(mean.reshape(28, 28).astype('uint8'))
    #tmpim.save(path + 'mean.jpg')
    pic = raw_pic - mean 
    cov = pic.T.dot(pic) / num_x
    lamda, v = np.linalg.eig(cov)
    lamda = np.real(lamda)
    v = np.real(v)
    index = lamda.argsort()[-feature:][::-1]
    sub_v = v.T[index]
    pic_pca = sub_v.dot(pic.T)
    pic_re = sub_v.T.dot(pic_pca).T + mean
    psnr_value = 0
    for i in range(num_x):
        psnr_value += psnr(pic_re[i], raw_pic[i], 28)
        if(i % 500 == 0):
            plt.imshow(pic_re[i].reshape(28, 28), cmap='gray')
            plt.savefig(path + str(int(i/500)) + '.jpg')
    return psnr_value / num_x

if mode == 'point':
    point()
if mode == 'mnist':
    '''
    X = []
    Y = []
    for i in range(1, 600, 10):
        feature = i
        X.append(i)
        Y.append(mnist())
    plt.plot(X, Y)
    plt.show()
    '''
    print(mnist())
```