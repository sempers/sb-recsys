from recsys_diploma import PredictiveRecommenderSystem
import datetime

products_csv_path = r"D:\books\google_drive\data\recsys_diploma\products.csv"
transactions_csv_path = r"D:\books\google_drive\data\recsys_diploma\transactions.csv"

recommender = PredictiveRecommenderSystem()

def test_learning():
    now = datetime.datetime.now()    
    recommender.load_data(products_csv_path, transactions_csv_path)
    recommender.learn_items()
    recommender.learn_users()
    delta = (datetime.datetime.now() - now).total_seconds()
    with open("time_test.txt", "a+") as f:
        f.write(f"Loading & learning time: {delta} seconds\n")
        print(f"Loading & learning time: {delta} seconds")

def test_prediction():
    now = datetime.datetime.now()
    predictions = recommender.predict_users_items(recommender.UIDS, 10)
    delta = (datetime.datetime.now() - now).total_seconds()
    with open("time_test.txt", "a+") as f:
        f.write(f"Predicting all time: {delta} seconds\n")
        print(f"Predicting all time: {delta} seconds")

def test_time():
    now = datetime.datetime.now()
    print((datetime.datetime.now() - now).total_seconds())

test_learning()
test_prediction()
#test_time()