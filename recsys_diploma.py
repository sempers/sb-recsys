import numpy as np
import pandas as pd
from collections import Counter

class ItemData:
    def __init__(self, aisle_id, dpt_id):
        self.aisle = aisle_id
        self.department = dpt_id
        self.global_count = 0
        self.global_p = 0.0        
        self.dpt_p = 0.0
        self.als_p = 0.0

class UserData:
    def __init__(self):
        self.orders = [] #заказы в виде списков product_id
        self.item_p = {} #априорная вероятность появления продуктов у пользователя по всей истории
        self.dpt_p = {} #усредненная вероятность появления категории у пользователя в заказе
        self.asl_p = {} #усредненная вероятность появления департамента у пользователя в заказе

class PredictiveRecommenderSystem:

    def __init__(self):
        self.AISLE_ITEMS = {}
        self.AISLE_ITEMS_PROBA = {}
        self.DEPARTMENT_ITEMS = {}
        self.user_data = {}
        self.item_data = {}        
        self.UIDS = []
        self.IIDS = []
        self.UIDS_index = {}
        self.UIDS_inverse_index = {}
        self.IIDS_index = {}
        self.IIDS_inverse_index = {}
        self.pr_df = None
        self.tx_df = None
        self.MAX_AISLE_POPULAR = 10
        print("[__init__] Recommender created")
        

    def load_data(self, products_csv_path, tx_csv_path):
        """
            Первичная загрузка датафреймов, построение простых словарей
            products_csv_path  - путь к файлу products.csv
            tx_csv_path        - путь к файлу transactions.csv
        """
        print("[load_data] Data started loading")
        self.pr_df = pd.read_csv(products_csv_path)
        self.tx_df = pd.read_csv(tx_csv_path)
        self.UIDS = sorted(self.tx_df['user_id'].unique())
        self.IIDS = sorted(self.pr_df['product_id'].unique())
        for _, row in self.pr_df.iterrows():
            self.item_data[row.product_id] = ItemData(row.aisle_id, row.department_id)
        self.DEPARTMENT_ITEMS = {dpt_id : self.pr_df[self.pr_df["department_id"] == dpt_id]["product_id"].values for dpt_id in self.pr_df["department_id"].unique()}
        self.AISLE_ITEMS = {aisle_id : self.pr_df[self.pr_df["aisle_id"] == aisle_id]["product_id"].values for aisle_id in self.pr_df["aisle_id"].unique()}
        print("[load_data] Data loaded")


    def learn_items(self):
        """
            Предобработка (обучение) информации по продуктам
        """
        print("[learn_items] Started learning")
        tx = self.tx_df
        #Считаем глобальную вероятность появления продукта в заказе P(I), а заодно глобальное число появлений каждого продукта
        orders_num = len(tx['order_id'].unique())
        for i, row in tx.groupby("product_id").agg({"order_id":"count"}).reset_index().iterrows():
            self.item_data[row.product_id].global_p = row.order_id / orders_num
            self.item_data[row.product_id].global_count = row.order_id
        
        #Считаем условную вероятность вхождения продукта I при департаменте D P(I | D)
        for _, iids in self.DEPARTMENT_ITEMS.items():
            #Общее число появлений всех продуктов всех департаментов
            sum = np.sum(self.item_data[iid].global_count for iid in iids)
            for iid in iids:
                self.item_data[iid].dpt_p = self.item_data[iid].global_count / sum

        #То же для продукта I и категории A P(I | A)
        for _, iids in self.AISLE_ITEMS.items():
            #Общее число появлений всех продуктов всех категорий
            sum = np.sum([self.item_data[iid].global_count for iid in iids])
            for iid in iids:
                self.item_data[iid].asl_p = self.item_data[iid].global_count / sum

        print("[learn_items] Items learned")

    
    def learn_users(self):
        """
            Предобработка (обучение) информации по пользователям
        """
        print("[learn_users] Started learning")
        tx = self.tx_df
        #Построение для извлечения информации по юзерам
        user_dict = tx.sort_values(["order_number", "add_to_cart_order"]).groupby(["user_id", "order_number"])\
            .agg({"product_id": lambda x: list(x)}).groupby(level=0).apply(lambda df: df.xs(df.name).to_dict()).to_dict()

        #Вспомогательные методы
        #Маппинг продукт -> категория для заказа
        def to_aisles(o):
            return [self.item_data[iid].aisle for iid in o]

        #Маппинг продукт -> департамент для заказа
        def to_departments(o):
            return [self.item_data[iid].department for iid in o]

        #Нормализация частот, посчитанных в Counter
        def norm_counter(counter, n):
            d = dict(counter.most_common())
            for k, _ in d.items():
                d[k] /= n
            return d

        #Нормализация частот, посчитанных в словаре
        def norm_proba(d, n):
            for k, _ in d.items():
                d[k] /= n
            return d
        
        #Расчет вероятности департамента/категории в заказе
        def update_proba(d, o):
            counter = Counter()
            counter.update(o)
            d1 = norm_counter(counter, len(o))
            for k, _ in d1.items():
                if k in d:
                    d[k] += d1[k]
                else:
                    d[k] = d1[k]
            return d
        for uid, df_udata in user_dict.items():
            udata = UserData()
            counter_items = Counter()
            aisles_p = {}
            dpts_p = {}
            sum = 0
            n = 0
            for _, order in df_udata['product_id'].items():
                udata.orders.append(order)
                counter_items.update(order)
                aisles_p = update_proba(aisles_p, to_aisles(order))
                dpts_p = update_proba(dpts_p, to_departments(order))
                sum += len(order)
                n += 1

            udata.item_p = norm_counter(counter_items, sum) 
            udata.asl_p = norm_proba(aisles_p, n) 
            udata.dpt_p = norm_proba(dpts_p, n)
            self.user_data[uid] = udata

        print("[learn_users] Users learned")

    #Наивный алгоритм
    def rank_items_for_user_naive(self, uid, k):
        return [x[0] for x in sorted(self.user_data[uid].item_p.items(), key=lambda x: -x[1])][:k]

    #Наивный алгоритм + Байес
    def rank_items_for_user(self, uid, k):
        udata = self.user_data[uid]
        #Начальное приближение - вероятности, зафиксированные в истории покупок
        target_proba = udata.item_p.copy()
        #Для каждой категории, зафиксированной у пользователя
        for aid, p_a in udata.asl_p.items():
            if aid not in self.AISLE_ITEMS_PROBA:
                #Строим список условных вероятностей P(I | A) и отбираем MAX_AISLE_POPULAR наибольших
                a_i_p = { iid: self.item_data[iid].asl_p for iid in self.AISLE_ITEMS[aid]}
                self.AISLE_ITEMS_PROBA[aid] = dict(sorted(a_i_p.items(), key=lambda x: -x[1])[:self.MAX_AISLE_POPULAR])
            #Добавляем в целевой список по формуле Байеса
            for iid, p_i_a in self.AISLE_ITEMS_PROBA[aid].items():
                #Если зафиксированная вероятность - пропускаем!
                if iid in target_proba: continue
                # P(I | Ui) = P(I | A)*P(A | Ui)
                target_proba[iid] = p_i_a * p_a

        return [x[0] for x in sorted(target_proba.items(), key=lambda x: -x[1])][:k]
    
    def predict_user_items(self, uid, k, naive_algo=True):
        """
            Возвращает список из k id релевантных продуктов для пользователя с id=uid
        """
        if naive_algo:
            return self.rank_items_for_user_naive(uid, k)
        else:
            return self.rank_items_for_user(uid, k)
    
    def predict_users_items(self, uids, k, naive_algo=True):
        """
            Возвращает список списков из k id релевантных продуктов для пользователей из списка uids
        """
        print("[predict_users_items] Started prediction")
        return [self.predict_user_items(uid, k, naive_algo=naive_algo) for uid in uids]         
