import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Loader:
    def __init__(self, selected_para):
        self.top_path = '../DB/CNS/Training_DB/'
        self.selected_para = pd.read_csv(selected_para)
        self.selected_cont_para = ''
        self.scaler = MinMaxScaler()
        self.time_col = 'KCNTOMS'

    def selected_cont_para_path(self, path):
        self.selected_cont_para = pd.read_csv(path)

    def show_file_list(self):
        return os.listdir(self.top_path)

    def read_dict_from_db_file(self, file_name):
        # csv 또는 pkl 파일을 읽어서 시나리오 명과 pandas Dataframe을 반환함
        return {'Se': file_name, 'db':pd.read_pickle(self.top_path +'\\' + file_name)}

    def show_property(self, file):
        return [print(_, file[_].keys()) for _ in file]

    def extracting_para_from_csv(self, file):
        # csv 파일내의 선택된 파라메터만 변수 정리
        # file : {'Se': .. , 'train_x_db': ..} -> {'Se': .. , 'train_x_db': 선택된 파라} + min_max scaler
        file['time'] = file['db'][self.time_col]
        file['train_x_db'] = file['db'][self.selected_para['PARA'].tolist()]
        file['train_x_db'] = file['train_x_db'].replace('---', 0)
        print(file['Se'], len(file['train_x_db'].min().to_numpy()), len(file['train_x_db'].max().to_numpy()))
        try:
            self.scaler.partial_fit([file['train_x_db'].min().to_numpy(), file['train_x_db'].max().to_numpy()])
        except:
            print('Error: 스케일러의 초기 길이가 맞지 않음 - 변수 갯수 문제')
        return file

    def getting_transferred_min_max_db(self, file_list, want_minmax=True):
        # get selected para db + min max scaler
        db = {_: self.extracting_para_from_csv(self.read_dict_from_db_file(_)) for _ in file_list}
        # transfer all db by using min max scaler  # pd.Dataframe -> numpy
        if want_minmax:
            for _ in db.keys():
                db[_]['train_x_db'] = self.scaler.transform(db[_]['train_x_db'].to_numpy())
        return db

    def classifier(self, file, max_accident_nub, accident_nub, mal_time):
        y_normal, y_mal = np.zeros((max_accident_nub)), np.zeros((max_accident_nub))
        y_normal[0], y_mal[accident_nub] = 1, 1

        def nor_mal(iter_time, time, y_normal, y_mal):
            if iter_time < time: return y_normal
            else: return y_mal

        file['train_y_db'] = [nor_mal(_, mal_time, y_normal, y_mal) for _ in file['time']]
        return file

    def extracting_cont_para_from_csv(self, file):
        file['train_y_db'] = file['db'][self.selected_cont_para['PARA'].tolist()]
        file['train_y_db'] = file['train_y_db'].to_numpy()
        return file

    def stack_db(self, file, time_leg):
        file['train_x_db'] = [file['train_x_db'][line:line+time_leg] for line in range(len(file['time'])
                                                                                       - time_leg + 1)]
        file['train_y_db'] = [file['train_y_db'][line + time_leg - 1] for line in range(len(file['time'])
                                                                                        - time_leg + 1)]
        return file

    def stack_cont_db(self, file, time_leg):
        file['train_x_db'] = [file['train_x_db'][line:line+time_leg] for line in range(len(file['time'])
                                                                                       - time_leg + 1)]
        file['train_y_db'] = [file['train_y_db'][line+time_leg] for line in range(len(file['time']) - time_leg + 1)]
        return file





