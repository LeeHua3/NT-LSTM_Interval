#指标参数
ft3_index=0
ft4_index=1
tsh_index=2

#数据参数
time_step=7
feature=3


#模型参数
#original
train_epochs=400
#train_epochs=800
train_lr=0.0002
train_decay=0.001

# train_epochs=800
# train_lr=0.002
# train_decay=0.01


#损失函数
n_ = 100 # batch size
lambda_ = 0.01  # lambda in loss fn
alpha_ = 0.05  # capturing (1-alpha)% of samples
soften_ = 160.
beta = 0.5
gama = 0.1


