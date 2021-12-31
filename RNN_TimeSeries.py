# Aishwarya Singh

import numpy as nup
import pandas as panda
import math
import sklearn
import sklearn.preprocessing
import matplotlib.pyplot as plotmat
from sklearn.metrics import mean_squared_error as mse
from math import sqrt as sq

#Downloading the Dataset
!wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fsly0Mw5pmYv2qAnARAOyF04W11wZbj9' -O prices-split-adjusted.csv

# https://drive.google.com/file/d/1fsly0Mw5pmYv2qAnARAOyF04W11wZbj9/view?usp=sharing

dframe = panda.read_csv("prices-split-adjusted.csv", index_col = 0 ,nrows=200000)

dframe.info()
dframe.head()


dframe.tail()

dframe.describe()

dframe.isnull().sum()

dframe.info()

plotmat.figure(figsize=(25, 5))
plotmat.subplot(1,2,1)
plotmat.title('stock price')
plotmat.xlabel('stock')
plotmat.ylabel('price')
plotmat.legend(loc='best')
plotmat.plot(dframe[dframe.symbol == 'WLTW'].open.values, color='red', label='open')
plotmat.plot(dframe[dframe.symbol == 'WLTW'].close.values, color='violet', label='close')
plotmat.plot(dframe[dframe.symbol == 'WLTW'].low.values, color='yellow', label='low')
plotmat.plot(dframe[dframe.symbol == 'WLTW'].high.values, color='black', label='high')

plotmat.subplot(1,2,2)
plotmat.plot(dframe[dframe.symbol == 'WLTW'].volume.values, color='blue', label='volume')
plotmat.title('stock volume')
plotmat.xlabel('stock')
plotmat.ylabel('volume')
plotmat.legend(loc='best')


def data_norm(dataf):
    mscale = sklearn.preprocessing.MinMaxScaler()
    dataf['open'] = mscale.fit_transform(dataf.open.values.reshape(-1,1))
    dataf['high'] = mscale.fit_transform(dataf.high.values.reshape(-1,1))
    dataf['low'] = mscale.fit_transform(dataf.low.values.reshape(-1,1))
    dataf['close'] = mscale.fit_transform(dataf['close'].values.reshape(-1,1))
    return dataf

testsizepercentage = 20 
def stock_data(data_s, seqsize):
    r_stock = data_s.to_numpy() 
    dstock = []

    for ir in range(len(r_stock) - seqsize): 
        dstock.append(r_stock[ir: seqsize+ir])
    
    dstock = nup.array(dstock);
    
    test_set = int(nup.round(testsizepercentage/100*dstock.shape[0]));
    train_set = dstock.shape[0] - (test_set);
    
    x_train = dstock[:train_set,:-1,:]
    y_train = dstock[:train_set,-1,:]
    
    x_test = dstock[train_set:,:-1,:]
    y_test = dstock[train_set:,-1,:]
   
    return [x_train, y_train, x_test, y_test]

sequencelength = 25

stockprice = dframe[dframe.symbol == 'WLTW'].copy()
stockprice.drop(['symbol'],1,inplace=True)
stockprice.drop(['volume'],1,inplace=True)

stockpricenormalized = stockprice.copy()
stockpricenormalized = data_norm(stockpricenormalized)

x_train, y_train, x_test, y_test = stock_data(stockpricenormalized, sequencelength)

print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ',y_test.shape)


plotmat.figure(figsize=(25,5))
plotmat.plot(stockpricenormalized.open.values, color='yellow', label='open')
plotmat.plot(stockpricenormalized.close.values, color='pink', label='close')
plotmat.plot(stockpricenormalized.low.values, color='green', label='low')
plotmat.plot(stockpricenormalized.high.values, color='blue', label='high')
plotmat.title('Normalized values')
plotmat.xlabel('stock')
plotmat.ylabel('normalized price')
plotmat.legend(loc='best')
plotmat.show()

def sigmoid_function(x):
    return 1 / (1 + nup.exp(-x))

dframe.describe()

tune_param = 0.00005 #learning_rate
min_value = -80.680000
max_value = 135.410004	 
cycle =  10           
seq = 24
LayerHidden = 100     
outputdim = 1
tuncatebt = 10


weightu = nup.random.uniform(0, 1, (LayerHidden, seq))
weightw = nup.random.uniform(0, 1, (LayerHidden, LayerHidden))
weightv = nup.random.uniform(0, 1, (outputdim, LayerHidden))


def fwd_pass_loss(x1,y1):
    loss = 0.0
    for rg in range(y1.shape[0]):
        x, y = x1[rg], y1[rg]                   
        p_act = nup.zeros((LayerHidden, 1))   
        for j in range(seq):
            n_io = nup.zeros(x.shape)    
            n_io[j] = x[j]              
            mu = nup.dot(weightu, n_io)
            mw = nup.dot(weightw, p_act)
            sum_m = mw + mu
            sum_m=nup.array(sum_m, dtype=nup.float128)
            sig = sigmoid_function(sum_m)
            mv = nup.dot(weightv, sig)
            p_act = sig
        lossrecord = (y - mv)**2 / 2
        loss += lossrecord
    loss = loss / float(y.shape[0])
    return loss

for ep in range(cycle):

    loss=fwd_pass_loss(x_train,y_train) 
    testloss=fwd_pass_loss(x_test,y_test)

    print('Iteration: ', ep + 1)
    print('test loss: ',testloss)
    print('Loss: ', loss)   
    
    for r in range(y_train.shape[0]):
        x, y = x_train[r], y_train[r]
        levels = []
        
        p_act = nup.zeros((LayerHidden, 1))
        dervdu = nup.zeros(weightu.shape)
        dervdv = nup.zeros(weightv.shape)
        dervdw = nup.zeros(weightw.shape)

        dervdu_t = nup.zeros(weightu.shape)
        dervdv_t = nup.zeros(weightv.shape)
        dervdw_t = nup.zeros(weightw.shape)
        
        dervdu_i = nup.zeros(weightu.shape)
        dervdw_i = nup.zeros(weightw.shape)
 
        for ts in range(seq):
            n_io = nup.zeros(x.shape)
            n_io[ts] = x[ts]
            mlu = nup.dot(weightu, n_io)
            mlw = nup.dot(weightw, p_act)
            sum_m = mlw + mlu
            sum_m=nup.array(sum_m, dtype=nup.float128)
            s = sigmoid_function(sum_m)
            mlv = nup.dot(weightv, s)
            levels.append({'s':s, 'p_act':p_act})
            p_act = s
            dmlv = (mlv - y)

        for td in range(seq):
            dervdv_t = nup.dot(dmlv, nup.transpose(levels[td]['s']))
            dsv = nup.dot(nup.transpose(weightv), dmlv)
            ds = dsv
            dadd = sum_m * (1 - sum_m) * ds 
            dmlw = dadd * nup.ones_like(mlw)
            dprev_s = nup.dot(nup.transpose(weightw), dmlw)

            for j in range(td-1, max(-1, td-tuncatebt-1), -1):
                ds = dsv + dprev_s
                dadd = sum_m * (1 - sum_m) * ds

                dmlw = dadd * nup.ones_like(mlw)
                dmlu = dadd * nup.ones_like(mlu)

                dervdw_i = nup.dot(weightw, levels[td]['p_act'])
                dprev_s = nup.dot(nup.transpose(weightw), dmlw)

                n_io = nup.zeros(x.shape)
                n_io[td] = x[td]
                dervdu_i = nup.dot(weightu, n_io)
                dx = nup.dot(nup.transpose(weightu), dmlu)
                dervdu_t[:, :dervdu_i.shape[1]] += dervdu_i
                dervdw_t[:, :dervdw_i.shape[1]] += dervdw_i
                
            dervdv += dervdv_t
            dervdu += dervdu_t
            dervdw += dervdw_t

            if dervdu.max() > max_value:
                dervdu[dervdu > max_value] = max_value
            if dervdv.max() > max_value:
                dervdv[dervdv > max_value] = max_value
            if dervdw.max() > max_value:
                dervdw[dervdw > max_value] = max_value
                
            
            if dervdu.min() < min_value:
                dervdu[dervdu < min_value] = min_value
            if dervdv.min() < min_value:
                dervdv[dervdv < min_value] = min_value
            if dervdw.min() < min_value:
                dervdw[dervdw < min_value] = min_value
        
        weightu -= tune_param * dervdu
        weightv -= tune_param * dervdv
        weightw -= tune_param * dervdw    


def fwd_pass_predict(x1,y1):
  preds = []
  for pr in range(y1.shape[0]):
    x, y = x1[pr], y1[pr]
    p_act = nup.zeros((LayerHidden, 1))
    for trq in range(seq):
        mlu = nup.dot(weightu, x)
        mlw = nup.dot(weightw, p_act)
        sum_m = mlw + mlu
        sum_m = nup.array(sum_m, dtype=nup.float128)
        s = sigmoid_function(sum_m)
        mlv = nup.dot(weightv, s)
        p_act = s
    preds.append(mlv)
  return preds

pred_train=fwd_pass_predict(x_train,y_train)
pred_train = nup.array(pred_train)

# print(pred_train)

plotmat.plot(pred_train[:, 0, 0], 'g')
plotmat.plot(y_train[:, 0], 'r')
plotmat.show()

rtmeansquare_train=sq(mse(y_train[:, 0],pred_train[:, 0, 0]))
print('RMSE Train',rtmeansquare_train)

pred_test=fwd_pass_predict(x_test,y_test)
pred_test = nup.array(pred_test)

plotmat.plot(pred_test[:, 0, 0], 'g')
plotmat.plot(y_test[:, 0], 'r')
plotmat.show()

rtmeansquare_test=sq(mse(y_test[:, 0],pred_test[:, 0, 0]))
print('RMSE Test',rtmeansquare_test)
