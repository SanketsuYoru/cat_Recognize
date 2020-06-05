import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import math as m



n=5000

costs = []
times=[]
for i in range(n):
    times.append(i+1)


def initPramaters(neurons,features):
    #这里实现只有一个隐藏层(n个神经元)的二分类神经网络
    #neurons：这一层的神经元个数
    #features：输入集合的特征点
    w_hidden1=np.mat(np.random.randn(neurons,features))*0.01#隐藏层1权重
    b_hidden1=np.zeros([neurons,1])#隐藏层1偏差

    w_output=np.mat(np.random.randn(1,neurons))*0.01#输出层权重
    b_output=np.zeros([1,1])#输出层偏差

    print("w_hidden1.shape"+str(w_hidden1.shape))
    print("b_hidden1.shape"+str(b_hidden1.shape))
    print("w_output.shape"+str(w_output.shape))
    print("b_output.shape"+str(b_output.shape))
    print("w_hidden1初始值")
    print(w_hidden1)
    return w_hidden1,b_hidden1,w_output,b_output




def predict(w_hidden1,b_hidden1,w_output,b_output, X):
    
    m  = X.shape[1] #图片的数量
    Y_prediction = np.zeros((1,m)) 
    
    #计预测猫在图片中出现的概率
    #隐藏层a[1]正向
    A_1,Z_1=forward_hidden(dataset=X,weight=w_hidden1,bias=b_hidden1)
    # print("A_1.shape"+str(A_1.shape))
    # print(A_1)
    #输出层a[2]正向
    A_2,Z_2=forward_output(A_1,weight=w_output,bias=b_output)
    for i in range(A_2.shape[1]):
        #将概率a [0，i]转换为实际预测p [0，i]
        Y_prediction[0,i] = 1 if A_2[0,i] > 0.5 else 0
    #使用断言
    assert(Y_prediction.shape == (1,m))
    
    return Y_prediction


def update(loopTimes,studyRate,w_hidden1,b_hidden1,w_output,b_output):
    #权值和偏差初始化
    for i in range(loopTimes):
        #隐藏层a[1]正向
        A_1,Z_1=forward_hidden(dataset=trainSet_A0,weight=w_hidden1,bias=b_hidden1)
        # print("A_1.shape"+str(A_1.shape))
        # print(A_1)
        #输出层a[2]正向
        A_2,Z_2=forward_output(A_1,weight=w_output,bias=b_output)

        # print("A_2.shape"+str(A_2.shape))
        # print(A_2)




        lost=Lost(A_2,trainSet_y)
        # print("lost.shape"+str(lost.shape))
        # print(lost)
        cost= lost.mean()
        costs.append(cost)
        print("cost"+str(cost))




        # print(m.exp(cost))
        dw_hidden1,db_hidden1,dw_output,db_output=backword(A_1,A_2,Z_1,Z_2,w_hidden1,b_hidden1,w_output,b_output,cost)

        # print("w_hidden1.shape"+str(w_hidden1.shape))
        # print("dw_hidden1.shape"+str(dw_hidden1.shape))
        # print(dw_hidden1)
        w_hidden1=np.mat(np.asarray(w_hidden1)-np.asarray(dw_hidden1)*np.asarray(studyRate))
        b_hidden1=np.mat(np.asarray(b_hidden1)-np.asarray(db_hidden1)*np.asarray(studyRate))
        w_output=np.mat(np.asarray(w_output)-np.asarray(db_output)*np.asarray(studyRate))
        b_output=np.mat(np.asarray(b_output)-np.asarray(db_output)*np.asarray(studyRate))
        # print("w_hidden1更新值")
        # print(w_hidden1)

    return w_hidden1,b_hidden1,w_output,b_output


def backword(A_1,A_2,Z_1,Z_2,w_hidden1,b_hidden1,w_output,b_output,cost):
    dZ_2=A_2-trainSet_y
    dw_output=np.dot(dZ_2,A_1.T)/209
    db_output=np.sum(dZ_2,axis=1)/209
    #db_output=np.sum(dZ_2,axis=1,keepdims=True)
    #dg_1=getReLUDer(Z_1)
    dg_1=getReLUDer(A_1)
    # print("dg_1.shape"+str(dg_1.shape))
    # print(dg_1)
    temp=np.asarray(np.dot(w_output.T,dZ_2))
    # print("temp.shape"+str(temp.shape))
    dZ_1=temp*np.asarray(dg_1)
    dZ_1=np.mat(dZ_1)
    dw_hidden1=np.dot(dZ_1,trainSet_A0.T)/209
    db_hidden1=np.sum(dZ_1,axis=1)/209
    return dw_hidden1,db_hidden1,dw_output,db_output



def getReLUDer(Z_1):
    Z_1[Z_1 <= 0] = 0
    Z_1[Z_1 > 0] = 1
    return Z_1

def getsigmoidDer(A_1):
    
    return np.mat(np.asarray(A_1)*np.asarray((1-A_1)))




def Lost(y_hat,y ):
    y=np.asarray(y)
    y_hat=np.asarray(y_hat)
    # print("y_hat.shape :"+str(y_hat.shape))
    # print("y.shape :"+str(y.shape))
    return -np.mat((y*np.log(y_hat)+(1-y)*np.log((1-y_hat))))

def sigmoid(z):
    y_hat= 1 / (1 + np.exp(-z))
    return y_hat

def ReLU(z):
    # print("ReLU\n")
    # print(z)
    # print(np.maximum(0,z))
    return np.maximum(0,z)

def forward_hidden(dataset,weight,bias):
    z=np.dot(weight,dataset)+bias
    # print("hiddenz.shape :"+str(z.shape))
    # print(z)
    y_hat=ReLU(z)
    #y_hat=sigmoid(z)
    return y_hat,z

def forward_output(dataset,weight,bias):
    z=np.dot(weight,dataset)+bias
    # print("outputz.shape :"+str(z.shape))
    # print(z)
    y_hat=sigmoid(z)
    return y_hat,z








#导入数据集
train_set_x_orig , trainSet_y , test_set_x_orig , testSet_y , classes = load_dataset()
#初始化参数
w_hidden1,b_hidden1,w_output,b_output=initPramaters(4,12288)#4个神经元（一个隐藏层），每个样本12288个特征值





#train_set_x_orig 是一个维度为(n_x，x_px，y_px，3）的矩阵，即（特征值数量，横向像素，纵向像素，色彩通道数）这里为209*64*64*3
# print("train_set_x_orig 是一个"+str(train_set_x_orig.shape)+"的矩阵")
# m_train = train_set_y.shape[1] #训练集里图片的数量。
# m_test = test_set_y.shape[1] #测试集里图片的数量。
# num_px = train_set_x_orig.shape[1] #训练、测试集里面的图片的宽度和高度（均为64x64）。

trainSet_A0  = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T#先除255，回头再看有什么区别
testSet_X = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T/255
# print("trainSet_A0.shape"+str(trainSet_A0.shape))
# print(trainSet_A0)
# print("trainSet_y.shape"+str(trainSet_y.shape))
# print("testSet_X.shape"+str(testSet_X.shape))
# print("testSet_y.shape"+str(testSet_y.shape))

Rate=0.0005
w_hidden1,b_hidden1,w_output,b_output=update(loopTimes=n,studyRate=Rate,w_hidden1=w_hidden1,b_hidden1=b_hidden1,w_output=w_output,b_output=b_output)#梯度下降1000次，学习率0.5
#studyRate=0.0005 cost0.09858708665567048 times=5000
#studyRate=0.0008

Test_result=predict(w_hidden1,b_hidden1,w_output,b_output, testSet_X)
print("Test_Y")
print(testSet_y.shape)
print(testSet_y)
print("Test_result")
print(Test_result.shape)
print(Test_result)

train_result=predict(w_hidden1,b_hidden1,w_output,b_output, trainSet_A0)
print("Train_Y")
print(trainSet_y.shape)
print(trainSet_y)
print("Train_result")
print(train_result.shape)
print(train_result)

print("训练集准确性："  , format(100 - np.mean(np.abs(train_result - trainSet_y)) * 100) ,"%")
print("测试集准确性："  , format(100 - np.mean(np.abs(Test_result - testSet_y)) * 100) ,"%")

plt.plot(times,np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('times')
plt.title("Learning rate ="+str(Rate))
plt.show()