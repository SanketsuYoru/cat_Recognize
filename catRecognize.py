import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import math as m



n=4000
costs_trainSet = []
costs_testSet = []
accuracy=[]
times=[]
for i in range(n):
    times.append(i+1)


def initPramaters(features):
    #这里实现只有一个隐藏层(n个神经元)的二分类神经网络
    #neurons：这一层的神经元个数
    #features：输入集合的特征点


    w_hidden1=(np.mat(np.random.randn(128,features)))*np.sqrt(2/features)#隐藏层1权重
    b_hidden1=np.zeros([128,1])#隐藏层1偏差
    w_hidden2=(np.mat(np.random.randn(64,128)))*np.sqrt(2/128)#隐藏层2权重
    b_hidden2=np.zeros([64,1])#隐藏层2偏差

    w_hidden3=(np.mat(np.random.randn(32,64)))*np.sqrt(2/64)#隐藏层3权重
    b_hidden3=np.zeros([32,1])#隐藏层3偏差

    w_output=(np.mat(np.random.randn(1,32)))*np.sqrt(2/32)#输出层权重
    b_output=np.zeros([1,1])#输出层偏差

    print("w_hidden1.shape"+str(w_hidden1.shape))
    print("b_hidden1.shape"+str(b_hidden1.shape))
    print("w_hidden2.shape"+str(w_hidden2.shape))
    print("b_hidden2.shape"+str(b_hidden2.shape))
    print("w_hidden3.shape"+str(w_hidden3.shape))
    print("b_hidden3.shape"+str(b_hidden3.shape))
    print("w_output.shape"+str(w_output.shape))
    print("b_output.shape"+str(b_output.shape))
    return w_hidden1,b_hidden1,w_hidden2,b_hidden2,w_hidden3,b_hidden3,w_output,b_output




def predict(w_hidden1,b_hidden1,w_hidden2,b_hidden2,w_hidden3,b_hidden3,w_output,b_output, X,test):
    
    m  = X.shape[1] #图片的数量
    Y_prediction = np.zeros((1,m)) 
    
    #计预测猫在图片中出现的概率
    #隐藏层a[1]正向
    A_1,Z_1=forward_hidden(dataset=X,weight=w_hidden1,bias=b_hidden1)
    #隐藏层a[2]正向
    A_2,Z_2=forward_hidden(A_1,weight=w_hidden2,bias=b_hidden2)
    #隐藏层a[3]正向
    A_3,Z_3=forward_hidden(A_2,weight=w_hidden3,bias=b_hidden3)
    #输出层a[4]正向
    A_4,Z_4=forward_output(A_3,weight=w_output,bias=b_output)

    # print("A_4_testSet.shape"+str(A_4.shape))
    # print(A_4)
    # print("testSet_y.shape"+str(testSet_y.shape))
    # print(testSet_y)

        #print("cost"+str(cost))
    # print("A_4.shape"+str(A_4.shape))
    # print(A_4)
    for i in range(A_4.shape[1]):
        #将概率a [0，i]转换为实际预测p [0，i]
        Y_prediction[0,i] = 1 if A_4[0,i] > 0.5 else 0
    #使用断言
    assert(Y_prediction.shape == (1,m))
    if test==True:
        lost=Lost(A_4,testSet_y)
        # print("lost.shape"+str(lost.shape))
        # print(lost)
        cost= lost.mean()
        costs_testSet.append(cost)
        accuracy.append(format(100 - np.mean(np.abs(Y_prediction - testSet_y)) * 100))
    
    return Y_prediction,A_4


def update(loopTimes,studyRate,w_hidden1,b_hidden1,w_hidden2,b_hidden2,w_hidden3,b_hidden3,w_output,b_output):

    w1,b1,w2,b2,w3,b3,w4,b4=w_hidden1,b_hidden1,w_hidden2,b_hidden2,w_hidden3,b_hidden3,w_output,b_output
    #权值和偏差初始化
    for i in range(loopTimes):
        #隐藏层a[1]正向
        A_1,Z_1=forward_hidden(dataset=trainSet_A0,weight=w1,bias=b1)
        #隐藏层a[2]正向
        A_2,Z_2=forward_hidden(dataset=A_1,weight=w2,bias=b2)
        #隐藏层a[3]正向
        A_3,Z_3=forward_hidden(dataset=A_2,weight=w3,bias=b3)
        #输出层a[3]正向
        A_4,Z_4=forward_output(dataset=A_3,weight=w4,bias=b4)

        # print("A_2.shape"+str(A_2.shape))
        # print(A_2)

        lost=Lost(A_4,trainSet_y)
        # print("lost.shape"+str(lost.shape))
        # print(lost)
        cost= lost.mean()
        costs_trainSet.append(cost)
        print("cost"+str(cost))




        # print(m.exp(cost))
        dw_hidden1,db_hidden1,dw_hidden2,db_hidden2,dw_hidden3,db_hidden3,dw_output,db_output=backword(A_1,A_2,A_3,A_4,Z_1,Z_2,Z_3,Z_4,w1,b1,w2,b2,w3,b3,w4,b4,cost)

        # print("w_hidden1.shape"+str(w_hidden1.shape))
        # print("dw_hidden1.shape"+str(dw_hidden1.shape))
        # print(dw_hidden1)
        w1=np.mat(np.asarray(w1)-np.asarray(dw_hidden1)*np.asarray(studyRate))
        b1=np.mat(np.asarray(b1)-np.asarray(db_hidden1)*np.asarray(studyRate))
        w2=np.mat(np.asarray(w2)-np.asarray(dw_hidden2)*np.asarray(studyRate))
        b2=np.mat(np.asarray(b2)-np.asarray(db_hidden2)*np.asarray(studyRate))
        w3=np.mat(np.asarray(w3)-np.asarray(dw_hidden3)*np.asarray(studyRate))
        b3=np.mat(np.asarray(b3)-np.asarray(db_hidden3)*np.asarray(studyRate))
        w4=np.mat(np.asarray(w4)-np.asarray(dw_output)*np.asarray(studyRate))
        b4=np.mat(np.asarray(b4)-np.asarray(db_output)*np.asarray(studyRate))
        predict(w1,b1,w2,b2,w3,b3,w4,b4, testSet_X,True)
        # print("w_hidden1更新值")
        # print(w_hidden1)

    return w1,b1,w2,b2,w3,b3,w4,b4


def backword(A_1,A_2,A_3,A_4,Z_1,Z_2,Z_3,Z_4,w_hidden1,b_hidden1,w_hidden2,b_hidden2,w_hidden3,b_hidden3,w_output,b_output,cost):
    dZ_4=A_4-trainSet_y
    dw_output=np.dot(dZ_4,A_3.T)/209
    db_output=np.sum(dZ_4,axis=1)/209

    dA_3=np.dot(w_output.T,dZ_4)
    dg_3=getReLUDer(A_3)
    dZ_3=np.mat(np.asarray(dA_3)*np.asarray(dg_3))
    dw_hidden3=np.dot(dZ_3,A_2.T)/209
    db_hidden3=np.sum(dZ_3,axis=1)/209


    dA_2=np.dot(w_hidden3.T,dZ_3)
    dg_2=getReLUDer(A_2)
    dZ_2=np.mat(np.asarray(dA_2)*np.asarray(dg_2))
    dw_hidden2=np.dot(dZ_2,A_1.T)/209
    db_hidden2=np.sum(dZ_2,axis=1)/209

    dA_1=np.dot(w_hidden2.T,dZ_2)
    dg_1=getReLUDer(A_1)
    dZ_1=np.mat(np.asarray(dA_1)*np.asarray(dg_1))
    dw_hidden1=np.dot(dZ_1,trainSet_A0.T)/209
    db_hidden1=np.sum(dZ_1,axis=1)/209

    # #trainSet=209

    # dZ_2=A_2-trainSet_y
    # dw_output=np.dot(dZ_2,A_1.T)/209
    # db_output=np.sum(dZ_2,axis=1)/209
    # #db_output=np.sum(dZ_2,axis=1,keepdims=True)
    # #dg_1=getReLUDer(Z_1)
    # dg_1=getReLUDer(A_1)
    # # print("dg_1.shape"+str(dg_1.shape))
    # # print(dg_1)
    # temp=np.asarray(np.dot(w_output.T,dZ_2))
    # # print("temp.shape"+str(temp.shape))
    # dZ_1=temp*np.asarray(dg_1)
    # dZ_1=np.mat(dZ_1)
    # dw_hidden1=np.dot(dZ_1,trainSet_A0.T)/209
    # db_hidden1=np.sum(dZ_1,axis=1)/209
    return dw_hidden1,db_hidden1,dw_hidden2,db_hidden2,dw_hidden3,db_hidden3,dw_output,db_output



def getReLUDer(A):
    A[A <= 0] = 0
    A[A > 0] = 1
    return A

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

def loadImage(path):
    # 读取图片
    image = Image.open(path)

    # 显示图片
    #image.show() 
    data = np.array(image)

    return data




cat=loadImage("C:\\Users\\I1661\\Desktop\\00001100_003.jpg")
cat2=loadImage("C:\\Users\\I1661\\Desktop\\00000100_003.jpg")
cat3=loadImage("C:\\Users\\I1661\\Desktop\\00000100_005.jpg")
not_cat=loadImage("C:\\Users\\I1661\\Desktop\\notcat.jpg")

not_cat_testimg_matrix_x=not_cat.reshape(1, -1).T/255
not_cat_testimg_y=np.mat([0])

cat3_testimg_matrix_x=cat3.reshape(1, -1).T/255
cat3_testimg_y=np.mat([1])


cat2_testimg_matrix_x=cat2.reshape(1, -1).T/255
cat2_testimg_y=np.mat([1])

cat_testimg_matrix_x=cat.reshape(1, -1).T/255
cat_testimg_y=np.mat([1])



# print("testimg_matrix.shape"+str(testimg_matrix_x.shape))
# print("testimg_y.shape"+str(testimg_y.shape))
#导入数据集
train_set_x_orig , trainSet_y , test_set_x_orig , testSet_y , classes = load_dataset()
#初始化参数
w_hidden1,b_hidden1,w_hidden2,b_hidden2,w_hidden3,b_hidden3,w_output,b_output=initPramaters(12288)#4个神经元（一个隐藏层），每个样本12288个特征值





# #train_set_x_orig 是一个维度为(n_x，x_px，y_px，3）的矩阵，即（特征值数量，横向像素，纵向像素，色彩通道数）这里为209*64*64*3
# # print("train_set_x_orig 是一个"+str(train_set_x_orig.shape)+"的矩阵")
# m_train = train_set_y.shape[1] #训练集里图片的数量。
# m_test = test_set_y.shape[1] #测试集里图片的数量。
#num_px = train_set_x_orig.shape[1] #训练、测试集里面的图片的宽度和高度（均为64x64）。
print("train_set_x_orig.shape"+str(train_set_x_orig.shape))
t=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1)


trainSet_A0  = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T/255#先除255，回头再看有什么区别
testSet_X = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T/255
# # print("trainSet_A0.shape"+str(trainSet_A0.shape))
# # print(trainSet_A0)
# # print("trainSet_y.shape"+str(trainSet_y.shape))
print("testSet_X.shape"+str(testSet_X.shape))
print("testSet_y.shape"+str(testSet_y.shape))

Rate=0.0008
w_hidden1,b_hidden1,w_hidden2,b_hidden2,w_hidden3,b_hidden3,w_output,b_output=update(loopTimes=n,studyRate=Rate,w_hidden1=w_hidden1,b_hidden1=b_hidden1,w_hidden2=w_hidden2,b_hidden2=b_hidden2,w_hidden3=w_hidden3,b_hidden3=b_hidden3,w_output=w_output,b_output=b_output)
# #studyRate=0.0005 cost0.09858708665567048 times=5000
# #studyRate=0.0008



plt.plot(times,np.squeeze(costs_trainSet),label="costs_trainSet")
plt.plot(times,np.squeeze(costs_testSet),label="costs_testSet")
plt.ylabel('cost')
plt.xlabel('times')
plt.title("Learning rate ="+str(Rate))
plt.show()

plt.plot(times,np.squeeze(accuracy),label="Accuracy_test")
plt.ylabel('Accuracy')
plt.xlabel('times')
plt.title("Learning rate ="+str(Rate))
plt.show()

Test_result,predict_value_te=predict(w_hidden1,b_hidden1,w_hidden2,b_hidden2,w_hidden3,b_hidden3,w_output,b_output, testSet_X,True)
print("Test_Y")
print(testSet_y.shape)
print(testSet_y)
print("Test_result")
print(Test_result.shape)
print(Test_result)
print("predict_value_te.shape"+str(predict_value_te.shape))
print(predict_value_te)

train_result,predict_value_tr=predict(w_hidden1,b_hidden1,w_hidden2,b_hidden2,w_hidden3,b_hidden3,w_output,b_output, trainSet_A0,False)
print("Train_Y")
print(trainSet_y.shape)
print(trainSet_y)
print("Train_result")
print(train_result.shape)
print(train_result)
print("predict_value_tr.shape"+str(predict_value_tr.shape))
print(predict_value_tr)



print("训练集准确性："  , format(100 - np.mean(np.abs(train_result - trainSet_y)) * 100) ,"%")
print("测试集准确性："  , format(100 - np.mean(np.abs(Test_result - testSet_y)) * 100) ,"%")


final_result,predict_value_final=predict(w_hidden1,b_hidden1,w_hidden2,b_hidden2,w_hidden3,b_hidden3,w_output,b_output,cat_testimg_matrix_x,False)
print("predict_value_final.shape"+str(predict_value_final.shape))
print(predict_value_final)
print("final_result")
print(final_result)

# print("图1是猫："  , format(predict_value_final * 100) ,"%")

# final_result,predict_value_final=predict(w_hidden1,b_hidden1,w_hidden2,b_hidden2,w_hidden3,b_hidden3,w_output,b_output,not_cat_testimg_matrix_x,False)
# print("predict_value_final.shape"+str(predict_value_final.shape))
# print(predict_value_final)
# print("final_result")
# print(final_result)

# print("图2不是猫："  , format(predict_value_final* 100) ,"%")

# final_result,predict_value_final=predict(w_hidden1,b_hidden1,w_hidden2,b_hidden2,w_hidden3,b_hidden3,w_output,b_output,cat2_testimg_matrix_x,False)
# print("predict_value_final.shape"+str(predict_value_final.shape))
# print(predict_value_final)
# print("final_result")
# print(final_result)

# print("图3是猫："  , format(predict_value_final* 100) ,"%")

# final_result,predict_value_final=predict(w_hidden1,b_hidden1,w_hidden2,b_hidden2,w_hidden3,b_hidden3,w_output,b_output,cat3_testimg_matrix_x,False)
# print("predict_value_final.shape"+str(predict_value_final.shape))
# print(predict_value_final)
# print("final_result")
# print(final_result)

# print("图4是猫："  , format(predict_value_final* 100) ,"%")