import numpy as np

class MainNet:
    def __init__(self, net_type='DNN', input_pa=1, output_pa=1, time_leg=1, classify=False):
        self.net_type = net_type
        self.input_pa = input_pa
        self.output_pa = output_pa
        self.time_leg = time_leg
        self.classify = classify
        self.main_network = self.build_model(net_type=self.net_type, in_pa=self.input_pa,
                                             ou_pa=self.output_pa, time_leg=self.time_leg)

    def build_model(self, net_type='DNN', in_pa=1, ou_pa=1, time_leg=1):
        from keras.layers import Dense, Input, Conv1D, MaxPooling1D, LSTM, Flatten, SimpleRNN, GRU
        from keras.models import Model

        # import os
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # 8 16 32 64 128 256 512 1024 2048
        if net_type == 'DNN':
            state = Input(batch_shape=(None, in_pa))
            shared = Dense(32, input_dim=in_pa, activation='relu', kernel_initializer='glorot_uniform')(state)
            # shared = Dense(48, activation='relu', kernel_initializer='glorot_uniform')(shared)

        elif net_type == 'CNN' or net_type == 'LSTM' or net_type == 'RNN' or net_type == 'GRU' or net_type == 'CLSTM':
            state = Input(batch_shape=(None, time_leg, in_pa))
            if net_type == 'CNN':
                shared = Conv1D(filters=30, kernel_size=3, strides=1, padding='same')(state)
                shared = MaxPooling1D(pool_size=2)(shared)
                shared = Flatten()(shared)
                shared = Dense(64)(shared)

            elif net_type == 'LSTM':
                shared = LSTM(30, activation='relu')(state)
                shared = Dense(64)(shared)

            elif net_type == 'RNN':
                shared = SimpleRNN(77, activation='relu')(state)
                shared = Dense(64)(shared)

            elif net_type == 'GRU':
                shared = GRU(37, activation='relu')(state)
                shared = Dense(64)(shared)

            elif net_type == 'CLSTM':
                shared = Conv1D(filters=30, kernel_size=3, strides=1, padding='same')(state)
                shared = MaxPooling1D(pool_size=2)(shared)
                shared = LSTM(40)(shared)

        # ----------------------------------------------------------------------------------------------------
        # Common output network
        hidden = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(shared)
        if self.classify:
            fin_out = Dense(ou_pa, activation='softmax', kernel_initializer='glorot_uniform')(hidden)
            loss_function = 'categorical_crossentropy'
        else:
            fin_out = Dense(ou_pa, activation='sigmoid', kernel_initializer='glorot_uniform')(hidden)
            loss_function = 'mean_absolute_error'

        network = Model(inputs=state, outputs=fin_out)

        network.summary()

        network.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
        return network

    def load_model(self):
        # self.actor.load_weights("ROD_A3C_actor.h5")
        # self.critic.load_weights("ROD_A3C_cric.h5")
        pass