import pymongo
import pandas as pd
import os
import glob

# Mongodb에 연결하기
connection = pymongo.MongoClient('mongodb://127.0.0.1:27017/')

# simulator_name

Sim_nub = 4
Sim_name = {0: 'CNS', 1: 'MAAP', 2: 'KINGS', 3: 'MarsCode', 4:'HanbitSim'}

db = connection.AbnormalDB

# 데이터 베이스의 Collection에 Document 업로드
collection_ = db[Sim_name[Sim_nub]]
print(collection_)

if Sim_nub == 0 or Sim_nub == 1 or Sim_nub == 3 or Sim_nub == 4:
    print(glob.glob('./Mongo_DB_Client/DBTot/*.csv'))
    get_paths = glob.glob('./Mongo_DB_Client/DBTot/*.csv')
elif Sim_nub == 2:
    print(glob.glob('./Mongo_DB_Client/DBTot/*.pkl'))
    get_paths = glob.glob('./Mongo_DB_Client/DBTot/*.pkl')


for path_ in get_paths:

    print(path_.split("\\")[1][:-4])
    info_ = path_.split("\\")[1][:-4]

    Scenario_Name = info_.split('#')[0]
    Scenario_Case = info_.split('#')[1]

    if Sim_nub == 0: csv_db = pd.read_csv(path_)
    if Sim_nub == 1: csv_db = pd.read_csv(path_)
    if Sim_nub == 3: csv_db = pd.read_csv(path_)
    if Sim_nub == 4: csv_db = pd.read_csv(path_)

    if Sim_nub == 2: csv_db = pd.read_pickle(path_)

    for para_name in csv_db.columns[1:]:
        Document_ = {
            'Simulator_Name': Sim_name[Sim_nub],
            'Scenario_Name': Scenario_Name,
            'Scenario_Case': Scenario_Case,
            'Para_Name': f'{para_name}',
            'History': list(csv_db[para_name]),
        }
        collection_.insert(Document_)