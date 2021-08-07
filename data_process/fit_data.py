import numpy as np
import matplotlib.pyplot as plt

#返回拟合曲线对应的参数,以及从曲线上采样的点
def fit_data(t,y,message="demo"):
    # 定义x、y散点坐标
    x = t
    x = np.array(x)
    y = np.array(y)

    #暂时不需要处理缺失值，因为前六个月缺失值很少
    #x,y=get_clear_data(x,y)


    #print('y is :\n', y)

    # 用3次多项式拟合
    f1 = np.polyfit(x, y, 5)

    #n次多项式，返回n+1个参数
    #print('f1 is :\n', f1)

    #p1就是由n次多项式形成的一个函数
    p1 = np.poly1d(f1)
    #print('p1 is :\n', p1)

    #将x通过函数变为yvals
    yvals = p1(x)

    #生成前六个月中，若干个采样点的数据。
    # 此处的间隔，代表我们采样的间隔。间隔越小，计算相似度所用的点就越多。
    x_all=np.arange(0,6,0.1)
    y_all=p1(x_all)

    #y是真值，yvals是y对应的t，通过拟合曲线计算的值。y和yvals越接近，说明拟合得越好
    # print("id",message)
    # print("y_all",y_all)
    # print("y",y)
    # print('yvals is :', yvals)
    # #绘图
    # plot_fit_data(x,y,yvals,x_all,y_all,message)
    
    #我们需要的是拟合函数的参数
    return f1,y_all

#去除y中的-1及其对应的t
def get_clear_data(t,y):
    clear_t=[]
    clear_y=[]
    for i in range(len(y)):
        if(y[i]!=-1):
            clear_t.append(t[i])
            clear_y.append(y[i])
    clear_t=np.array(clear_t)
    clear_y=np.array(clear_y)

    return clear_t,clear_y

#绘制拟合效果
def plot_fit_data(x, y, yvals, x_all, y_all, message):
    plot1 = plt.plot(x, y, 's', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plot3 = plt.plot(x_all, y_all, 'b', label='y_all')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=4)  # 指定legend的位置右下角
    plt.title(message)
    plt.show()

if __name__=="__main__":

    t=[0,1,2,3,3.1,3.5,4,5.5]
    y=[-1,9.84,7.92,3.99,-1,4,3.91,-1]
    fit_data(t,y)
