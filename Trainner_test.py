import unittest
import pickle
from Trainner import trainner

class Trainner_test(unittest.TestCase):
    def test_train_LSTM_control(self):
        train = trainner()
        train.load_db(file='DB_MAKER/CNS_predict_abnormal_cont_train_db.bin')
        train.train_control(type_net='LSTM')

    def test_train_LSTM_clasiffer(self):
        train = trainner()
        train.load_db(file='DB_MAKER/CNS_predict_abnormal_accident_train_db.bin')
        train.train_clasiffer(type_net='LSTM')

    def test_train_CNN_control(self):
        train = trainner()
        train.load_db(file='DB_MAKER/CNS_predict_abnormal_cont_train_db.bin')
        train.train_control(type_net='CNN')

    def test_train_CNN_clasiffer(self):
        train = trainner()
        train.load_db(file='DB_MAKER/CNS_predict_abnormal_accident_train_db.bin')
        train.train_clasiffer(type_net='CNN')

    def test_train_RNN_control(self):
        train = trainner()
        train.load_db(file='DB_MAKER/CNS_predict_abnormal_cont_train_db.bin')
        train.train_control(type_net='RNN')

    def test_train_RNN_clasiffer(self):
        train = trainner()
        train.load_db(file='DB_MAKER/CNS_predict_abnormal_accident_train_db.bin')
        train.train_clasiffer(type_net='RNN')

    def test_train_CLSTM_control(self):
        train = trainner()
        train.load_db(file='DB_MAKER/CNS_predict_abnormal_cont_train_db.bin')
        train.train_control(type_net='CLSTM')

    def test_train_CLSTM_clasiffer(self):
        train = trainner()
        train.load_db(file='DB_MAKER/CNS_predict_abnormal_accident_train_db.bin')
        train.train_clasiffer(type_net='CLSTM')

    def test_train_GRU_control(self):
        train = trainner()
        train.load_db(file='DB_MAKER/CNS_predict_abnormal_cont_train_db.bin')
        train.train_control(type_net='GRU')

    def test_train_GRU_clasiffer(self):
        train = trainner()
        train.load_db(file='DB_MAKER/CNS_predict_abnormal_accident_train_db.bin')
        train.train_clasiffer(type_net='GRU')

    def test_all(self):
        self.test_train_CNN_clasiffer()
        self.test_train_GRU_clasiffer()
        self.test_train_LSTM_clasiffer()
        self.test_train_CLSTM_clasiffer()
        self.test_train_RNN_clasiffer()

    def test_cl_all(self):
        self.test_train_CNN_clasiffer()
        self.test_train_GRU_clasiffer()
        self.test_train_LSTM_clasiffer()
        self.test_train_CLSTM_clasiffer()
        self.test_train_RNN_clasiffer()

    def test_cont_all(self):
        self.test_train_CNN_control()
        self.test_train_GRU_control()
        self.test_train_LSTM_control()
        self.test_train_CLSTM_control()
        self.test_train_RNN_control()

    def test_al(self):
        from playsound import playsound
        playsound('al.MP3')

    def test_(self):
        self.test_cl_all()
        self.test_cl_all()
        self.test_cl_all()
        self.test_cl_all()
        self.test_cl_all()
        while True:
            self.test_al()

    def test_show_network_structure(self):
        train = trainner()
        train.load_db(file='DB_MAKER/CNS_predict_abnormal_accident_train_db.bin')
        train.show_network(type_net='GRU', classify=True)
        train.show_network(type_net='LSTM', classify=True)
        train.show_network(type_net='CLSTM', classify=True)
        train.show_network(type_net='RNN', classify=True)
        train.show_network(type_net='CNN', classify=True)

    def test_show_network_cont_structure(self):
        train = trainner()
        train.load_db(file='DB_MAKER/CNS_predict_abnormal_cont_train_db.bin')
        train.show_network(type_net='GRU', classify=False)
        train.show_network(type_net='LSTM', classify=False)
        train.show_network(type_net='CLSTM', classify=False)
        train.show_network(type_net='RNN', classify=False)
        train.show_network(type_net='CNN', classify=False)
