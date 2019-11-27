import numpy as np
import makedata as data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

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
    mnist = fetch_mldata('MNIST original', data_home='D:/LEARNING/CODES/ML/机器学习实验/PCA/data/')
    raw_pic = np.float64(mnist['data'][:num_x])
    path = 'D:/LEARNING/CODES/ML/机器学习实验/PCA/image/'
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