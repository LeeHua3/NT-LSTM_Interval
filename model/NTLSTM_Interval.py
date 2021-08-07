from params.parameters_Interval import *
import tensorflow as tf


class NTLSTM_Interval(object):
    def init_weights(self, input_dim, output_dim, name, std=0.1, reg=None):
        return tf.get_variable(name, shape=[input_dim, output_dim], initializer=tf.random_normal_initializer(0.0, std),
                               regularizer=reg)

    def init_bias(self, output_dim, name):
        return tf.get_variable(name, shape=[output_dim], initializer=tf.constant_initializer(1.0))

    def no_init_weights(self, input_dim, output_dim, name):
        return tf.get_variable(name, shape=[input_dim, output_dim])

    def no_init_bias(self, output_dim, name):
        return tf.get_variable(name, shape=[output_dim])


    #初值bias对训练非常重要！否则会训练失败！
    def init_Interval_bias(self, output_dim, name):
        return tf.get_variable(name, shape=[output_dim], initializer=tf.constant_initializer([5.0,3.0]))




    # LRH:understanded
    def __init__(self, o_input_dim, o_output_dim, o_hidden_dim, fc_dim, train):

        self.input_dim = o_input_dim
        self.hidden_dim = o_hidden_dim

        # [batch size x seq length x input dim]
        #ft3,ft4,tsh
        self.input = tf.placeholder('float', shape=[None, None, self.input_dim])

        #IP改1：修改被预测的输出值
        #self.labels = tf.placeholder('float', shape=[None, output_dim])
        #y上界、下界、点值
        self.y_ulp_true= tf.placeholder('float', shape=[None, 3])
        #IP改1/

        self.time = tf.placeholder('float', shape=[None, None])
        self.keep_prob = tf.placeholder(tf.float32)

        if train == 1:

            self.Wi = self.init_weights(self.input_dim, self.hidden_dim, name='Input_Hidden_weight', reg=None)
            self.Ui = self.init_weights(self.hidden_dim, self.hidden_dim, name='Input_State_weight', reg=None)
            self.bi = self.init_bias(self.hidden_dim, name='Input_Hidden_bias')

            self.Wf = self.init_weights(self.input_dim, self.hidden_dim, name='Forget_Hidden_weight', reg=None)
            self.Uf = self.init_weights(self.hidden_dim, self.hidden_dim, name='Forget_State_weight', reg=None)
            self.bf = self.init_bias(self.hidden_dim, name='Forget_Hidden_bias')

            self.Wog = self.init_weights(self.input_dim, self.hidden_dim, name='Output_Hidden_weight', reg=None)
            self.Uog = self.init_weights(self.hidden_dim, self.hidden_dim, name='Output_State_weight', reg=None)
            self.bog = self.init_bias(self.hidden_dim, name='Output_Hidden_bias')

            self.Wc = self.init_weights(self.input_dim, self.hidden_dim, name='Cell_Hidden_weight', reg=None)
            self.Uc = self.init_weights(self.hidden_dim, self.hidden_dim, name='Cell_State_weight', reg=None)
            self.bc = self.init_bias(self.hidden_dim, name='Cell_Hidden_bias')

            self.W_decomp = self.init_weights(self.hidden_dim, self.hidden_dim, name='Decomposition_Hidden_weight',
                                              reg=None)
            self.b_decomp = self.init_bias(self.hidden_dim, name='Decomposition_Hidden_bias_enc')

            #IP改2.1：指定输出层权重
            #I表示Interval，P表示Point,表明输出层同时输出点估计和区间估计
            self.W_output_IP = self.init_weights(self.hidden_dim, o_output_dim, name='Output_IP_weight',
                                                 reg=None)  # tf.contrib.layers.l2_regularizer(scale=0.001)
            #必须指定初始bias
            self.b_output_IP = self.init_Interval_bias(o_output_dim, name='Output_IP_bias')
            #IP改2.1/

            # self.Wo = self.init_weights(self.hidden_dim, fc_dim, name='Fc_Layer_weight',
            #                             reg=None)  # tf.contrib.layers.l2_regularizer(scale=0.001)
            # self.bo = self.init_bias(fc_dim, name='Fc_Layer_bias')

            # self.W_softmax = self.init_weights(fc_dim, output_dim, name='Output_Layer_weight',
            #                                    reg=None)  # tf.contrib.layers.l2_regularizer(scale=0.001)
            # self.b_softmax = self.init_bias(output_dim, name='Output_Layer_bias')

        else:

            self.Wi = self.no_init_weights(self.input_dim, self.hidden_dim, name='Input_Hidden_weight')
            self.Ui = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Input_State_weight')
            self.bi = self.no_init_bias(self.hidden_dim, name='Input_Hidden_bias')

            self.Wf = self.no_init_weights(self.input_dim, self.hidden_dim, name='Forget_Hidden_weight')
            self.Uf = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Forget_State_weight')
            self.bf = self.no_init_bias(self.hidden_dim, name='Forget_Hidden_bias')

            self.Wog = self.no_init_weights(self.input_dim, self.hidden_dim, name='Output_Hidden_weight')
            self.Uog = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Output_State_weight')
            self.bog = self.no_init_bias(self.hidden_dim, name='Output_Hidden_bias')

            self.Wc = self.no_init_weights(self.input_dim, self.hidden_dim, name='Cell_Hidden_weight')
            self.Uc = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Cell_State_weight')
            self.bc = self.no_init_bias(self.hidden_dim, name='Cell_Hidden_bias')

            self.W_decomp = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Decomposition_Hidden_weight')
            self.b_decomp = self.no_init_bias(self.hidden_dim, name='Decomposition_Hidden_bias_enc')

            #IP改2.2：指定输出层权重
            self.W_output_IP = self.no_init_weights(self.hidden_dim, o_output_dim, name='Output_IP_weight')
            self.b_output_IP = self.no_init_bias(o_output_dim, name='Output_IP_bias')
            # IP改2.2/

            # self.Wo = self.no_init_weights(self.hidden_dim, fc_dim, name='Fc_Layer_weight')
            # self.bo = self.no_init_bias(fc_dim, name='Fc_Layer_bias')

            # self.W_softmax = self.no_init_weights(fc_dim, output_dim, name='Output_Layer_weight')
            # self.b_softmax = self.no_init_bias(output_dim, name='Output_Layer_bias')



    # LRH: modified and checked:2020.12.20
    def NTLSTM_Unit(self, prev_hidden_memory, concat_input):
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)

        batch_size = tf.shape(concat_input)[0]

        x = tf.slice(concat_input, [0, 1], [batch_size, self.input_dim])
        t = tf.slice(concat_input, [0, 0], [batch_size, 1])



        # Input gate
        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(prev_hidden_state, self.Ui) + self.bi)

        # Forget Gate
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(prev_hidden_state, self.Uf) + self.bf)

        # Output Gate
        o = tf.sigmoid(tf.matmul(x, self.Wog) + tf.matmul(prev_hidden_state, self.Uog) + self.bog)

        # Candidate Memory Cell
        C = tf.nn.tanh(tf.matmul(x, self.Wc) + tf.matmul(prev_hidden_state, self.Uc) + self.bc)

        # Current Memory cell
        # lrh:* is elem_wise
        Ct = f * prev_cell + i * C

        # Dealing with time irregularity
        # Map elapse time in days or months
        # T=216*128,means each state of 128 states should be discounted
        T = self.map_elapse_time(t)

        # Decompose the previous cell if there is a elapse time
        C_ST = tf.nn.tanh(tf.matmul(Ct, self.W_decomp) + self.b_decomp)
        C_ST_dis = tf.multiply(T, C_ST)
        # if T is 0, then the weight is one
        #Ct_star is Ct*
        Ct_star = Ct - C_ST + C_ST_dis


        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(Ct_star)

        return tf.stack([current_hidden_state, Ct_star])

    #LRH:understanded
    def get_states(self):  # Returns all hidden states for the samples in a batch
        batch_size = tf.shape(self.input)[0]
        scan_input_ = tf.transpose(self.input, perm=[2, 0, 1])
        scan_input = tf.transpose(scan_input_)  # scan input is [seq_length x batch_size x input_dim]
        scan_time = tf.transpose(self.time)  # scan_time [seq_length x batch_size]
        initial_hidden = tf.zeros([batch_size, self.hidden_dim], tf.float32)
        ini_state_cell = tf.stack([initial_hidden, initial_hidden])
        # make scan_time [seq_length x batch_size x 1]
        scan_time = tf.reshape(scan_time, [tf.shape(scan_time)[0], tf.shape(scan_time)[1], 1])
        concat_input = tf.concat([scan_time, scan_input], 2)  # [seq_length x batch_size x input_dim+1]
        packed_hidden_states = tf.scan(self.NTLSTM_Unit, concat_input, initializer=ini_state_cell, name='states')
        all_states = packed_hidden_states[:, 0, :, :]
        # 打印
        #all_states = tf.Print(all_states, ['all_states: ', all_states])
        return all_states

    # for one sample
    # def get_output(self, state):
    #     output = tf.nn.relu(tf.matmul(state, self.Wo) + self.bo)
    #     output = tf.nn.dropout(output, self.keep_prob)
    #     output = tf.matmul(output, self.W_softmax) + self.b_softmax
    #     return output

    #IP改3：从hidden_dim映射到output_IP_dim,不使用激活函数，这就是输出层
    def get_output(self, state):
        output = tf.matmul(state, self.W_output_IP) + self.b_output_IP
        return output

    # for one batch
    def get_outputs(self):  # Returns all the outputs
        all_states = self.get_states()
        all_outputs = tf.map_fn(self.get_output, all_states)
        output = tf.reverse(all_outputs, [0])[0, :, :]  # Compatible with tensorflow 1.2.1
        # output = tf.reverse(all_outputs, [True, False, False])[0, :, :] # Compatible with tensorflow 0.12.1
        # 打印
        #output = tf.Print(output, ['output: ', output])
        return output

    # def get_cost_acc(self):
    #     logits = self.get_outputs()
    #     cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))
    #     y_pred = tf.argmax(logits, 1)
    #     y = tf.argmax(self.labels, 1)
    #     return cross_entropy, y_pred, y, logits, self.labels


    #IP改4：不再使用交叉熵损失预测类别，改为使用自定义的QD和QD-TI预测真值
    def get_cost_acc(self):
        #logits = self.get_outputs()
        y_ulp_pred=self.get_outputs()
        #打印
        #y_ulp_pred=tf.Print(y_ulp_pred, ['y_ulp_pred: ', y_ulp_pred])
        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))
        #Loss=qd_objective(self.y_ulp_true, y_ulp_pred)
        Loss = C_TI_QD_objective(self.y_ulp_true, y_ulp_pred)
        #Loss = rmse_objective(self.y_ulp_true, y_ulp_pred)

        # 打印
        #print("经过损失函数后：")
        #Loss = tf.Print(Loss, ['Loss: ', Loss])
        # self.y_ulp_true = tf.Print(self.y_ulp_true, ['self.y_ulp_true: ', self.y_ulp_true])
        #y_ulp_pred = tf.Print(y_ulp_pred, ['y_ulp_pred_after_loss: ', y_ulp_pred])

        return Loss, self.y_ulp_true, y_ulp_pred


    # LRH:understanded
    def map_elapse_time(self, t):

        c1 = tf.constant(1, dtype=tf.float32)
        c2 = tf.constant(2.7183, dtype=tf.float32)

        # T = tf.multiply(self.wt, t) + self.bt

        T = tf.div(c1, tf.log(t + c2), name='Log_elapse_time')

        Ones = tf.ones([1, self.hidden_dim], dtype=tf.float32)

        T = tf.matmul(T, Ones)

        return T

def rmse_objective(y_true, y_pred):

    y_t=y_true[:,2]
    y_p_u=y_pred[:,0]
    y_p_l = y_pred[:, 1]

    Loss_u=tf.reduce_mean(tf.sqrt(y_t-y_p_u))
    #Loss_u = tf.Print(Loss_u, ['Loss_u: ', Loss_u])
    Loss_l = tf.reduce_mean(tf.sqrt(y_t - y_p_l))
    Loss=Loss_u+Loss_l
    # 打印
    #Loss = tf.Print(Loss, ['rmse_Loss: ', Loss])

    return Loss


def qd_objective(y_true, y_pred):
    '''Loss_QD-soft, from algorithm 1'''
    #修改了y_true的定义，不再stack，所以不再取第0维度
    #y_true = y_true[:,0]
    y_u = y_pred[:,0]
    y_l = y_pred[:,1]

    K_HU = tf.maximum(0.,tf.sign(y_u - y_true[:,2]))
    K_HL = tf.maximum(0.,tf.sign(y_true[:,2] - y_l))
    K_H = tf.multiply(K_HU, K_HL)

    K_SU = tf.sigmoid(soften_ * (y_u - y_true[:,2]))
    K_SL = tf.sigmoid(soften_ * (y_true[:,2] - y_l))
    K_S = tf.multiply(K_SU, K_SL)

    MPIW_c = tf.reduce_sum(tf.multiply((y_u - y_l),K_H))/tf.reduce_sum(K_H)
    PICP_H = tf.reduce_mean(K_H)
    PICP_S = tf.reduce_mean(K_S)

    Loss_S = MPIW_c + lambda_ * n_ / (alpha_*(1-alpha_)) * tf.maximum(0.,(1-alpha_) - PICP_S)

    return Loss_S

def C_TI_QD_objective(y_true, y_pred):
    #修改了y_true的定义，不再stack，所以不再取第0维度
    #y_true = y_true[:,0]
    y_u = y_pred[:,0]
    y_l = y_pred[:,1]

    K_HU = tf.maximum(0.,tf.sign(y_u - y_true[:,2]))
    K_HL = tf.maximum(0.,tf.sign(y_true[:,2] - y_l))
    K_H = tf.multiply(K_HU, K_HL)

    K_SU = tf.sigmoid(soften_ * (y_u - y_true[:,2]))
    K_SL = tf.sigmoid(soften_ * (y_true[:,2] - y_l))
    K_S = tf.multiply(K_SU, K_SL)

    MPIW_c = tf.reduce_sum(tf.multiply((y_u - y_l),K_H))/tf.reduce_sum(K_H)
    PICP_H = tf.reduce_mean(K_H)
    PICP_S = tf.reduce_mean(K_S)

    PICP_SEC=lambda_ * n_ / (alpha_*(1-alpha_)) * tf.maximum(0.,(1-alpha_) - PICP_S)


    u_sub = y_pred[:, 0] - y_true[:, 0]
    l_sub = y_pred[:, 1] - y_true[:, 1]

    # rmse
    #Interval_loss = gama*tf.sqrt(tf.reduce_mean(tf.square(l_sub) + tf.square(u_sub)))
    # mse
    Interval_loss = gama*tf.reduce_mean(tf.square(l_sub) + tf.square(u_sub))

    # 区间宽度不影响Interval_loss的数量级。
    Interval_loss = tf.Print(Interval_loss, ['Interval_loss: ', Interval_loss])
    PICP_SEC = tf.Print(PICP_SEC, ['PICP_SEC: ', PICP_SEC])
    MPIW_c =tf.Print(MPIW_c, ['MPIW_c: ', MPIW_c])

    Loss_S = MPIW_c + PICP_SEC + Interval_loss

    return Loss_S