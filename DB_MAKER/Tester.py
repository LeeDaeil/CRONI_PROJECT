import unittest
import pickle
from DB_MAKER.File_loader import Loader


class Loader_test(unittest.TestCase):
    def test_show_file_list(self):
        # 디폴트
        print(Loader(selected_para='selected_para.csv').show_file_list())

    def test_show_file_list_other_path(self):
        # 파일 로더 위치 변경 시
        file_loader = Loader(selected_para='selected_para.csv')
        file_loader.top_path = '../DB/KINGS/Training_DB/'
        print(file_loader.show_file_list())

    def test_read_dict_from_db_file(self):
        # 파일리스트 중 1개의 데이터의 Key값 읽기
        file_loader = Loader(selected_para='selected_para.csv')
        file_list = file_loader.show_file_list()
        print(file_loader.read_dict_from_db_file(file_list[0]))

    def test_extracting_para_from_csv(self):
        # 파라메터 추출하기
        loader = Loader(selected_para='selected_para.csv')
        file_list = loader.show_file_list()
        db = loader.read_dict_from_db_file(file_list[1])
        print(loader.extracting_para_from_csv(db))

    def test_extracting_para_from_csv_selected_time(self):
        # 다른 시뮬레이터 데이터를 사용할때 time_col의 기준을 변경해하는 경우
        loader = Loader(selected_para='selected_para_KINGS.csv')
        loader.top_path = '../DB/KINGS/Training_DB/'
        loader.time_col = 'Time'
        file_list = loader.show_file_list()
        db = loader.read_dict_from_db_file(file_list[1])
        print(loader.extracting_para_from_csv(db))

    def test_2_file_extracting_para_from_csv(self):
        # 파일 2개에서 데이터 정렬
        loader = Loader(selected_para='selected_para.csv')
        file_list = loader.show_file_list()
        db = {_: loader.extracting_para_from_csv(loader.read_dict_from_db_file(_)) for _ in file_list[0:2]}
        loader.show_property(db)
        print(db, type(db))
        ###---- 여기까지 작업함.

    def test_show_property(self):
        loader = Loader(selected_para='selected_para.csv')
        file_list = loader.show_file_list()
        db = {_: loader.extracting_para_from_csv(loader.read_dict_from_db_file(_)) for _ in file_list[0:2]}
        loader.show_property(db)

    def test_getting_transferred_min_max_db(self):
        loader = Loader(selected_para='selected_para.csv')
        file_list = loader.show_file_list()
        db = loader.getting_transferred_min_max_db(file_list[0:10])
        loader.show_property(db)
        # [print(_) for _ in db[0]['db'][0]]
        print('done')

        with open('min_max_scaler.bin', 'wb') as f:
            pickle.dump(loader.scaler, f)
        print('Save scaler')

    def test_load_scaler(self):
        with open('min_max_scaler.bin', 'rb') as f:
            scaler = pickle.load(f)
        print(scaler, scaler.data_min_, scaler.data_max_)
        # print(f'Max: {scaler.data_min_.shape()}, Min: {scaler.data_max_.shape()}')

    def test_classifier(self):
        loader = Loader(selected_para='selected_para.csv')
        file_list = loader.show_file_list()
        db = loader.getting_transferred_min_max_db(file_list[0:3])
        loader.show_property(db)
        db[0] = loader.classifier(db['ab15-07_1001.pkl'], max_accident_nub=3, accident_nub=1, mal_time=30)
        print(db[0], type(db))

    def test_stack_db(self):
        loader = Loader(selected_para='selected_para.csv')
        file_list = loader.show_file_list()
        db = loader.getting_transferred_min_max_db(file_list[0:1])
        db['ab15-07_1001.pkl'] = loader.classifier(db['ab15-07_1001.pkl'], max_accident_nub=3, accident_nub=1, mal_time=30)
        db['ab15-07_1001.pkl'] = loader.stack_db(db['ab15-07_1001.pkl'], 10)
        print(db['ab15-07_1001.pkl'])

    def test_stack_db_with_for(self):
        loader = Loader(selected_para='selected_para.csv')
        file_list = loader.show_file_list()
        db = loader.getting_transferred_min_max_db(file_list[0:3])
        controller = [
            ['ab15-07_1001.pkl', 1], ['ab15-07_1002.pkl', 2], ['ab15-07_1003.pkl', 2]
        ]
        for se, ac_nub in controller:
            db[se] = loader.classifier(db[se], max_accident_nub=3, accident_nub=ac_nub, mal_time=30)
            db[se] = loader.stack_db(db[se], 10)
        [print(db[_]['Se']) for _ in db.keys()]
        loader.show_property(db)

    def test_make_train_db(self):
        loader = Loader(selected_para='selected_para.csv')
        file_list = loader.show_file_list()
        db = loader.getting_transferred_min_max_db(file_list)
        controller = [
            ['ab15-07_1001.pkl', 1], ['ab15-07_1002.pkl', 2], ['ab15-07_1003.pkl', 2]
        ]
        for se, ac_nub in controller:
            db[se] = loader.classifier(db[se], max_accident_nub=len(db), accident_nub=ac_nub, mal_time=30)
            db[se] = loader.stack_db(db[se], 10)
        [print(db[_]['Se']) for _ in db.keys()]
        loader.show_property(db)

        # save db
        with open('train_db.bin', 'wb') as f:
            pickle.dump(db, f)

        # load db
        # with open('train_db.bin', 'rb') as f:
        #     db_temp = pickle.load(f)

# ---------------------------------------------------------------------------------------------------------------------
# 훈련에 사용될 데이터 생성하는 파트
# ---------------------------------------------------------------------------------------------------------------------
    def test_work_make_train_db(self):
        loader = Loader(selected_para='selected_para_predict_state.csv')
        file_list = loader.show_file_list()
        db = loader.getting_transferred_min_max_db(file_list)
        controller = []
        for file_name in file_list:
            if file_name[:7] == 'ab21-01': controller.append([file_name, 1])
            elif file_name[:7] == 'ab21-02': controller.append([file_name, 2])
            elif file_name[:7] == 'ab20-01': controller.append([file_name, 3])
            elif file_name[:7] == 'ab20-04': controller.append([file_name, 4])
            elif file_name[:7] == 'ab15-07': controller.append([file_name, 5])
            elif file_name[:7] == 'ab15-08': controller.append([file_name, 6])
            elif file_name[:7] == 'ab63-04': controller.append([file_name, 7])
            elif file_name[:7] == 'ab63-02': controller.append([file_name, 8])
            elif file_name[:7] == 'ab63-03': controller.append([file_name, 9])
            elif file_name[:7] == 'ab21-12': controller.append([file_name, 10])
            elif file_name[:7] == 'ab19-02': controller.append([file_name, 11])
            elif file_name[:7] == 'ab21-11': controller.append([file_name, 12])
            elif file_name[:7] == 'ab23-03': controller.append([file_name, 13])
            elif file_name[:7] == 'ab80-02': controller.append([file_name, 14])
            elif file_name[:7] == 'ab60-02': controller.append([file_name, 15])
            elif file_name[:7] == 'ab59-02': controller.append([file_name, 16])
            elif file_name[:7] == 'ab23-01': controller.append([file_name, 17])
            elif file_name[:7] == 'ab23-06': controller.append([file_name, 18])
            elif file_name[:7] == 'ab59-01': controller.append([file_name, 19])
            elif file_name[:7] == 'ab64-03': controller.append([file_name, 20])

        for se, ac_nub in controller:
            db[se] = loader.classifier(db[se], max_accident_nub=21, accident_nub=ac_nub, mal_time=30)
            db[se] = loader.stack_db(db[se], 10)
        [print(db[_]['Se']) for _ in db.keys()]

        for _ in db.keys():
            del db[_]['db']

        loader.show_property(db)

        # save db
        with open('CNS_predict_abnormal_accident_train_db.bin', 'wb') as f:
            pickle.dump(db, f)

        # save scaler
        with open('min_max_scaler.bin', 'wb') as f:
            pickle.dump(loader.scaler, f)

    def test_work_make_cont_train_db(self):
        loader = Loader(selected_para='selected_para_predict_state.csv')
        # loader.selected_cont_para_path('selected_para_predict_cond_letdown_flow.csv')
        loader.selected_cont_para_path('selected_para_CRONI.csv')
        file_list = loader.show_file_list()
        db = loader.getting_transferred_min_max_db(file_list)
        controller = []
        for file_name in file_list:
            if file_name[:7] == 'ab21-01': controller.append([file_name, 1])
            elif file_name[:7] == 'ab21-02': controller.append([file_name, 2])
            elif file_name[:7] == 'ab20-01': controller.append([file_name, 3])
            elif file_name[:7] == 'ab20-04': controller.append([file_name, 4])
            elif file_name[:7] == 'ab15-07': controller.append([file_name, 5])
            elif file_name[:7] == 'ab15-08': controller.append([file_name, 6])
            elif file_name[:7] == 'ab63-04': controller.append([file_name, 7])
            elif file_name[:7] == 'ab63-02': controller.append([file_name, 8])
            elif file_name[:7] == 'ab63-03': controller.append([file_name, 9])
            elif file_name[:7] == 'ab21-12': controller.append([file_name, 10])
            elif file_name[:7] == 'ab19-02': controller.append([file_name, 11])
            elif file_name[:7] == 'ab21-11': controller.append([file_name, 12])
            elif file_name[:7] == 'ab23-03': controller.append([file_name, 13])
            elif file_name[:7] == 'ab80-02': controller.append([file_name, 14])
            elif file_name[:7] == 'ab60-02': controller.append([file_name, 15])
            elif file_name[:7] == 'ab59-02': controller.append([file_name, 16])
            elif file_name[:7] == 'ab23-01': controller.append([file_name, 17])
            elif file_name[:7] == 'ab23-06': controller.append([file_name, 18])
            elif file_name[:7] == 'ab59-01': controller.append([file_name, 19])
            elif file_name[:7] == 'ab64-03': controller.append([file_name, 20])

        for se, _ in controller:
            db[se] = loader.extracting_cont_para_from_csv(db[se])
            db[se] = loader.stack_db(db[se], 10)
        [print(db[_]['Se']) for _ in db.keys()]

        for _ in db.keys():
            del db[_]['db']

        loader.show_property(db)

        # save db
        with open('CNS_predict_abnormal_cont_train_db.bin', 'wb') as f:
            pickle.dump(db, f)

        # save scaler
        with open('min_max_scaler.bin', 'wb') as f:
            pickle.dump(loader.scaler, f)
