import numpy as np
import matplotlib.pyplot as plt

def my_plot(y_true,y_pred,inf):
    # 只显示30个样本方便观察
    y_true = y_true[:50]
    y_pred = y_pred[:50]

    #为每个样本生成0~sample_num-1的横坐标
    temp_shape=y_true.shape[0]
    x_index=np.arange(0,temp_shape)

    # bound true
    #u_y_true=y_true[:,0]
    #l_y_true=y_true[:,1]
    #plt.scatter(x_index, u_y_true, color='r',s=1)  # upper boundary prediction
    #plt.scatter(x_index, l_y_true, color='g',s=1)  #  lower boundary prediction


    #区间估计：y_true有两个维度
    if(y_pred.shape[1]==2):
        # point true
        p_y_true=y_true[:,2]
        plt.scatter(x_index, p_y_true, color='b', s=1)

        # bound pred
        y_u_pred = y_pred[:, 0]
        y_l_pred = y_pred[:, 1]
        y_m_pred = (y_u_pred+y_l_pred)/2

        plt.plot(x_index, y_u_pred, color='r')  # upper boundary prediction
        plt.plot(x_index, y_l_pred, color='g')  #  lower boundary prediction
        plt.plot(x_index, y_m_pred, color='y')  #  mid prediction
    #点估计：y_true只有一个维度
    else:
        plt.scatter(x_index, y_true, color='b', s=1)
        plt.plot(x_index, y_pred, color='c')


    plt.title(inf)
    plt.xlim(-1, temp_shape)
    mean=np.mean(y_true)
    #all_index：为了适配所有指标，修改显示范围
    plt.ylim(-3, 4*mean)
    plt.show()

