import pymongo

# Mongodb에 연결하기
connection = pymongo.MongoClient('mongodb://127.0.0.1:27017/')
print(connection)

# 데이터 베이스에 접근
print(connection.AbnormalDB)
db = connection.AbnormalDB

# 데이터 베이스의 Collection에 Document 업로드
Document_ = {
    'Name': "CNS",
    'History': [0, 1, 2, 3, 4, 5, 5],
    'Type': 'float'
}
collection_ = db.Abnormal_DB2
print(collection_)
# collection_.insert(Document_)
print(collection_.find())

# Collection에서 Document 얻기
print(collection_.find_one())
print(collection_.find_one({'Name': 'Lee'}))


for doc_ in collection_.find({'Name': 'Lee'}):
    print(doc_)

# 특정 값만 얻기
for doc_ in collection_.find({'Name': 'Lee'}, {'_id': 0, 'History': 1}):
    # List 값인 History 값을 받아서 프린트
    print(doc_['History'], type(doc_['History']))

print('=' * 10)
# 2가지 Name 값 얻기
for doc_ in collection_.find({'Name': {'$in': ['Lee', 'Lee2']}}, {'_id': 0, 'History': 1}):
    # List 값인 History 값을 받아서 프린트
    print(doc_['History'], type(doc_['History']))