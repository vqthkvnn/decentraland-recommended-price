import json
import math
from datetime import datetime
import numpy as np
import xgboost as xgb
from keras import Sequential
from keras.layers import Bidirectional, Dense, LSTM
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


def avg_psi(parcels):
    if len(parcels) == 0:
        return 0, 0
    sumx = 0
    sumy = 0
    for value_ in parcels:
        sumx += value_[0]
        sumy += value_[1]
    return int(sumx / len(parcels)), int(sumy / len(parcels))


def belong_to_psi(point, parcels):
    for value_ in parcels:
        if value_[0] == point[0] and value_[1] == point[1]:
            return True
    return False


def location_potential(point, estates):
    score = 0.0
    for value_ in estates.keys():
        x, y = avg_psi(estates[value_]["child"])
        if x == 0 and y == 0:
            continue
        else:
            blp = belong_to_psi(point, estates[value_]["child"])
            if far_from_point_to_estate(point, (x, y)) is True:
                if estates[value_]["name"] in ["Genesis Plaza", "Pixel Plaza 1", "Pixel Plaza 2"]:
                    score += 0.5
                else:
                    score += 0.3
            if blp:
                score += 0.5

    return score / 2.2


def far_from_point_to_estate(position_point, position_estate):
    return math.sqrt(math.pow((position_point[0] - position_estate[0]), 2) +
                     math.pow((position_point[1] - position_estate[1]), 2)) <= 10


def check_ranger_index(index):
    if index < -150:
        return -150
    if index > 150:
        return 150
    return index


def road_near_point(data, point):
    "matrix 5*5 point near point"
    count = 0
    price_near = 0
    sum_ = 0
    for i in range(check_ranger_index(point[0] - 4), check_ranger_index(point[0] + 5)):
        for j in range(check_ranger_index(point[1] - 4), check_ranger_index(point[1] + 5)):
            if data[str(i)][str(j)]["now_price"] is not None:
                price_near += data[str(i)][str(j)]["now_price"]
                sum_ += 1
            if data[str(i)][str(j)]["estate"] is not None and data[str(i)][str(j)]["estate"]["nft"] is not None:
                if data[str(i)][str(j)]["nft"]["name"] is None and \
                        data[str(i)][str(j)]["estate"]["nft"]["name"] == "Roads":
                    count += 1.21

    if sum_ == 0:
        return count / 2.2, 0
    return count / 2.2, float(price_near / sum_)


input_estate = open('estates.json', encoding="utf8")
json_estate = json.load(input_estate)
estate_list = {}
for estate_ in json_estate["data"]["estates"]:
    estate_list[estate_["id"]] = {
        "name": estate_["data"]["name"],
        "child": []
    }
    for parcels_ in estate_["parcels"]:
        estate_list[estate_["id"]]["child"].append((int(parcels_["x"]), int(parcels_["y"])))
input_estate.close()
print("Done load estate data")
input_file = open('data.json', encoding="utf8")
json_array = json.load(input_file)

data_ = {}
for idx, item in enumerate(json_array):
    if str(item["x"]) not in data_:
        data_[str(item["x"])] = {}
    dt = {
        "estate": item["estate"],
        "id": item["id"],
        "nft": item["nft"]
    }
    now_price = 0
    now_time = None
    min_price_buy = 0
    price_want_sell = []
    avg_min_price = {"total": 0, "count": 0}
    if len(item["nft"]["orders"]) > 0:
        for index, value in enumerate(item["nft"]["orders"]):
            if value['status'] == "sold" and now_price == 0:
                now_price = int(value["price"]) / float(math.pow(10, 19))
                now_time = datetime.utcfromtimestamp(float(value["updatedAt"]))
                price_want_sell.append(int(value["price"]) / float(math.pow(10, 19)))
            else:
                if value['status'] == "sold" and now_price != 0:
                    price_want_sell.append(int(value["price"]) / float(math.pow(10, 19)))
    if len(item["nft"]["bids"]) > 0:
        has_update = False
        for index, value in enumerate(item["nft"]["bids"]):
            if now_price < int(value["price"]) / float(math.pow(10, 19)) and value['status'] == "sold" \
                    and has_update is False:
                if now_time is None or now_time < datetime.utcfromtimestamp(float(value["updatedAt"])):
                    now_price = int(value["price"]) / float(math.pow(10, 19))
                    now_time = datetime.utcfromtimestamp(float(value["updatedAt"]))
                    has_update = True
            if min_price_buy < int(value["price"]) / float(math.pow(10, 19)) and value['status'] == "open":
                # first value for open
                min_price_buy = int(value["price"]) / float(math.pow(10, 19))
            else:
                avg_min_price["total"] += int(value["price"]) / float(math.pow(10, 19))
                avg_min_price["count"] += 1
    avg_buy = 0
    cstn = 0
    if min_price_buy > 0 and avg_min_price["total"] > 0:
        avg_buy = ((avg_min_price["total"] + min_price_buy) / float((avg_min_price["count"] + 1)))
        cstn = float(min_price_buy / avg_buy)

    if now_price > 0 and avg_buy > 0:
        cstn = float(now_price / avg_buy)
    dt["now_price"] = int(now_price)
    dt["cstn"] = cstn * 100
    if min_price_buy == 0:
            dt["min_price_buy"] = int(now_price)
    else:
        dt["min_price_buy"] = min_price_buy

    dt["avg_price_sell"] = avg_buy
    data_[str(item["x"])][str(item["y"])] = dt
print("Done load and process data")
input_file.close()
data_train = []
label_train = []
for i in data_.keys():
    for j in data_[i].keys():
        nr, prn = road_near_point(data_, (int(i), int(j)))
        tnvt = location_potential((int(i), int(j)), estate_list)
        data_[i][j]['csvt'] = nr
        data_[i][j]["tnvt"] = tnvt
        data_[i][j]["prn"] = prn
print("Done set value")
min_pl = []
for i in data_.keys():
    for j in data_[i].keys():
        if data_[i][j]["now_price"] == 0:
            continue
        else:
            val = [data_[i][j]["cstn"], data_[i][j]["csvt"], data_[i][j]["tnvt"], data_[i][j]["min_price_buy"],
                   data_[i][j]["avg_price_sell"]]
            min_pl.append(data_[i][j]["min_price_buy"])
            data_train.append(np.array(val))
            label_train.append(data_[i][j]["now_price"])
print("Done convert data")
# import pandas as pd

data_train = np.array(data_train)
# pd.DataFrame(data_train).to_csv("file_2.csv")
# np.savetxt("file.csv", data_train, delimiter=",")
label_train = np.array(label_train)
# np.savetxt("lb.csv", label_train, delimiter=",")
# pd.DataFrame(label_train).to_csv("lkb.csv")
n_features = 5
n_steps = 1
X = data_train.reshape((data_train.shape[0], 1, n_features))
model = Sequential()
model.add(Bidirectional(LSTM(196, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
H = model.fit(X, label_train, epochs=200, verbose=1, validation_data=(X, label_train))
model.save("predict.model")
yhat = model.predict(X, verbose=0)
rmse = math.sqrt(mean_squared_error(label_train, yhat))
print('Value RMSE: %.3f' % rmse)
plt.figure()
plt.plot(label_train[10:30], color='red', label='now price')
plt.plot(yhat[10:30], color='green', label='price predict')
plt.plot(min_pl[10:30], color='yellow', label='min price')
plt.xlabel("Number of sample")
plt.ylabel("price (coin)/10")
plt.show()
plt.savefig("pred.png")
