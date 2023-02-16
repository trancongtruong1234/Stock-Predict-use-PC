import json
from django.shortcuts import HttpResponse, render, redirect
from ast import main
import numpy as np
import pandas as pd
import math
from datetime import datetime
from dateutil.parser import parse
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae
import timeit

file_data = ''


def home(request):
    return render(request, "app/index.html")


def add(request):
    num1 = int(request.POST["num1"])
    num2 = int(request.POST["num2"])
    num3 = request.POST["num3"]
    num4 = int(request.POST["comboboxPriceIndex"])

    link = request.FILES["file"]

    n3 = num3.split(":")
    test= int(n3[0])
    train= int(n3[1])
    rate = test/(test+train)

    df,RMSE, MAE, MAPE,timeTrain,timeTest = main(link, num1, num2, rate, num4)
    timeALL = timeTrain+timeTest

    result = df.values.tolist()

    request.session["num1"] = num1
    request.session["num2"] = num2
    request.session["num3"] = num3
    request.session["num4"] = num4
    request.session["result"] = result
    request.session["RMSE"] = RMSE
    request.session["MAE"] = MAE
    request.session["MAPE"] = MAPE
    request.session["timeTrain"] = timeTrain
    request.session["timeTest"] = timeTest
    request.session["timeALL"] = timeALL
    return redirect("home")


def handleFileUpload(request):
    global file_data
    file = request.FILES["file"]
    file_data = pd.DataFrame(pd.read_csv(file))
    json_stuff = json.dumps(
        {"priceIndex": list(file_data.columns.values.tolist())})
    return HttpResponse(json_stuff, content_type="application/json")


def showTrainTest(request):
    train_test = request.POST["train_test"]
    priceIndex = int(request.POST["priceIndex"])
    n3 = train_test.split(":")
    testIndex= int(n3[0])
    trainIndex= int(n3[1])
    rate = testIndex/(testIndex+trainIndex)

    train = getTrainData(file_data,rate, priceIndex)
    test = file_data.drop(train.index)

    json_stuff = json.dumps({"train_row": len(train), "train_column": len(
        train.columns), "test_row": len(test), "test_column": len(test.columns)})
    return HttpResponse(json_stuff, content_type="application/json")


def Model(d, t, data, v):
    X, T = convertData(d, t, data, v)
    M = poly(X)
    s = 0
    _w = []
    _tmp = []

    # tinh w
    for i in range(len(M)):
        s += M[i].dot(M[i])

    for i in range(t):
        _tmp = []
        for j in range(len(M)):
            _tmp.append(M[j] * T[j][i] * (1 / s))
        _w.append(_tmp)
    return _w


def predict(w, d, t, data, priceIndex):
    result = []
    X, T = convertData(d, t, data, priceIndex)
    # print(X)
    # print(T)
    for item in X:
        tmp = []
        pz = twoOrderPoly(item)
        for i in range(t):
            sum = 0
            for px in w[i]:
                sum += px.dot(pz)
            tmp.append(sum)
        result.append(tmp)
    return np.array(result), T


def getLastResult(d, t, data, dayIndex, T, resultPre):
    D = []
    for i in range(data[data.columns[0]].count() - d - t + 1):
        # for i in range (250):
        tmpD = []
        for k in range(0, t):
            tmpD.append(data.iloc[i + d + k][dayIndex])
        D.append(np.array(tmpD))
    D = np.array(D)

    dt = pd.DataFrame(
        {
            "Day": D.flatten(),
            "predictPrice": resultPre.flatten(),
            "realPrice": T.flatten(),
        }
    )
    #dt['Day'] = parse(dt['Day'])
    dt = dt.groupby(['Day']).mean().reset_index().sort_values(by=['Day'])
    # print('Day:')
    # print(D)
    # print('predict')
    # print(resultPre)
    # print('real')
    # print(T)
    return dt


def convertData(d, t, data, v):
    X = []
    T = []
    for i in range(data[data.columns[0]].count() - d - t + 1):
        # for i in range (250):
        tmpX = []
        tmpT = []
        for j in range(0, d):
            tmpX.append(data.iloc[i + j][v])
        X.append(np.array(tmpX))
        for k in range(0, t):
            tmpT.append(data.iloc[i + d + k][v])
        T.append(np.array(tmpT))
    X = np.array(X)
    T = np.array(T)
    return X, T


def poly(X):
    M = []
    tmp = []
    for x in X:
        tmp = twoOrderPoly(x)
        M.append(tmp)
    M = np.array(M)
    # print(M)
    return M


def twoOrderPoly(X):
    m = [1]
    tmp = []
    l = len(X)
    for i in range(l - 1):
        m.append(X[i])
        tmp.append(X[i])
        m.append(pow(X[i], 2))
        m.extend(np.array(tmp) * X[i + 1])
    m.append(X[l - 1])
    m.append(pow(X[l - 1], 2))
    return m

def caculaRMSE(dt):
    MSE = np.square(np.subtract(np.array(dt.iloc[:,[1]]),np.array(dt.iloc[:,[2]]))).mean() 
    RMSE = math.sqrt(MSE)
    return RMSE

def getDayIndex(data):
    for i in range(len(data.iloc[1])-1):
        try:
            parse(data.iloc[1][i])
            return i
        except:
            continue
    return -1
def getTrainData(data, k, priceIndex):
    lenData = len(data) #chieu dai data
    l = int((lenData*k)/3) # 3*l = chieu dai tap train, lay 3 tap hop lai thanh tap train
    smax = -999999 #tong cua doan du lieu trong thoi ki tang truong
    smin = 999999 #tong cua doan du lieu trong thoi ki suy thoai

    lmax =0
    lmin =0
    imax =0
    imin = 0
    iavg =0
    lavg =0
    listOfs =[]
 
    #tim doan du lieu thoi ki tang va giam
    for i in range(1,lenData - l,l):
        s = 0
        for j in range(i, i+l):
            s += (data.iloc[i][priceIndex] - data.iloc[i-1][priceIndex])
        if(s > smax):
            smax =s
            imax = i
            lmax = i+l
        if(s < smin):
            smin =s
            imin =i
            lmin = i+l
        listOfs.append(s)

    # tim doan du lieu trong thoi ki on dinh
    smean = 999999 #dung de tim doan du lieu co do lech thap nhat
    m = np.mean(listOfs)
    j =0
    for i in range(1,lenData - l,l):
        s = abs(listOfs[j] - m)
        if(s < smean ):
            smean =s
            iavg = i
            lavg = i+l
        j+=1
    # print(listOfs)
    # print('mean',m)
    # print('group max', smax, imax, lmax)
    # print('group min', smin, imin, lmin)
    # print('group avg', smean, iavg, lavg)
    #print(data.iloc[range(imax,lmax),:])
    #print(data.iloc[range(imin,lmin),:])
    train = pd.concat([data.iloc[range(iavg,lavg),:],data.iloc[range(imax,lmax),:],data.iloc[range(imin,lmin),:]],ignore_index=True)
    return train
def caculaMAE(dt):
    MAE = mae(np.array(dt.iloc[:,[1]]),np.array(dt.iloc[:,[2]]))
    return MAE

def caculaMAPE(dt):
    MAPE = mape(np.array(dt.iloc[:,[1]]),np.array(dt.iloc[:,[2]]))
    return MAPE

def main(link, num1, num2, num3, num4):
    # link = r"C:\Users\thanh\Downloads\test.csv"
    startTrain = timeit.default_timer()
    data = pd.DataFrame(pd.read_csv(link))
    # print(data)
    d = num1
    t = num2
    k = num3
    priceIndex = num4
    dayIndex = getDayIndex(data)

    #train = data.iloc[range(0, int(len(data) * k)), :]
    train = getTrainData(data,k, priceIndex)
    test = data.drop(train.index)

    w = Model(d, t, train, priceIndex)
    stopTrain = timeit.default_timer()

    resultPre, T = predict(w, d, t, test, priceIndex)
    dt = getLastResult(d, t, test, dayIndex, T, resultPre)
    RMSE = caculaRMSE(dt)
    MAE = caculaMAE(dt)
    MAPE = caculaMAPE(dt)
    stopTest = timeit.default_timer()

    timeTrain = stopTrain - startTrain
    timeTest = stopTest - stopTrain
    #print(RMSE)
    print('timeTrain: ', timeTrain)
    print('time Test: ', timeTest)
    return dt,RMSE,MAE,MAPE,timeTrain, timeTest
