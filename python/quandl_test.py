import quandl as q
import pandas as pd
import numpy as np
import sys
import os
import TA as ta
from sklearn.neighbors import NearestNeighbors as nn
from sklearn.neighbors import KNeighborsRegressor as nnr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf

q.ApiConfig.api_key = "xRfZ7bPGqPgCyzxpHQzu"
mydata = q.get("BITFINEX/BTCUSD")
mydata.rename(columns={"Ask":"Close"}, inplace=True)
print mydata
#df = pd.DataFrame(mydata)

#df.to_csv(mydata("data/cme.csv"))
#print mydata

mydata.to_csv("data/cme.csv")

def plotequity(pred, res, title):
    mydpi = 96
    plt.close("all")
    pos =  np.where( pred > 0, 1, np.where( pred <-0, -1, 0 ) )
    eq = np.cumsum( pos * res )
    fig = plt.figure(figsize=(1800 / mydpi, 1000 / mydpi), dpi=mydpi)
    ax = fig.add_subplot(1)
    ax.set_xticklabels(res.index)
    ax.plot( eq.values  )
    ax.plot( np.cumsum(res).values  )
    fig.autofmt_xdate()
    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    plt.title(title)
    plt.show()

class dataset:
    i=0
    dataset_width = 0;
    dataset_length = 0;

    def __init__(self):
        i = 1

    def createfromohlcv(self, ohlcv):
        self.dataset = pd.DataFrame(ohlcv['Close'].diff(), columns=['Close'], index=ohlcv.index)
        self.dataset['MA'] = ta.MA(ohlcv, 10)-ohlcv['Close']
        self.dataset['MA2'] = ta.MA(ohlcv, 20)-ohlcv['Close']
        self.dataset['Last2'] = self.dataset['Close'].shift(1)
        self.dataset['Res'] = ohlcv['Close'].shift(-1).diff()
        self.dataset = self.dataset.dropna(axis=0)
        self.dataset_length = self.dataset.shape[0]
        self.dataset_width = self.dataset.shape[1]


    def normalize(self):
        for i in range(0, self.insample_data.shape[1]):
            cd = self.insample_data.iloc[:,i]
            mean = cd.mean()
            stddev = cd.std()
            self.insample_data.iloc[:, i] = (cd - mean) / stddev
            cd = self.learn_data.iloc[:,i]
            self.learn_data.iloc[:, i] = (cd - mean) / stddev
            cd = self.outofsample_data.iloc[:,i]
            self.outofsample_data.iloc[:, i] = (cd - mean) / stddev


    def getnames(self):
        return self.insample_data.columns.values

    def splitdata(self, insample_size, test_size, outofsample_size ):
        self.insamplelearn_items = np.zeros(self.dataset_length, dtype=bool)
        self.insampletest_items = np.zeros(self.dataset_length, dtype=bool)
        self.outofsample_items = np.zeros(self.dataset_length, dtype=bool)

        self.insamplelearn_items[:int(self.dataset_length*insample_size)]=True
        self.insampletest_items[int(self.dataset_length * insample_size): int(self.dataset_length * insample_size + self.dataset_length * test_size)] = True
        self.outofsample_items[int(self.dataset_length * insample_size+ self.dataset_length * test_size):] = True

        self.insample_data = self.dataset.iloc[self.insamplelearn_items, :self.dataset_width-1]
        self.insample_res = self.dataset.iloc[self.insamplelearn_items, (self.dataset_width-1):]
        self.learn_data = self.dataset.iloc[self.insampletest_items, :self.dataset_width-1]
        self.learn_res = self.dataset.iloc[self.insampletest_items, (self.dataset_width-1):]
        self.outofsample_data = self.dataset.iloc[self.outofsample_items, :self.dataset_width-1]
        self.outofsample_res = self.dataset.iloc[self.outofsample_items, (self.dataset_width-1):]
        return

    def selectedcolumns(self, dataset, columns):
        self.insample_data = dataset.insample_data.iloc[:,columns]
        self.learn_data = dataset.learn_data.iloc[:,columns]
        self.outofsample_data = dataset.outofsample_data.iloc[:, columns]
        self.insample_res = dataset.insample_res
        self.learn_res = dataset.learn_res
        self.outofsample_res = dataset.outofsample_res

class ai:
    def __init__(self, data, algo, par1, par2):
        self.dataset=data
        self.algo=algo
        self.par1=par1
        self.par2=par2
        if( self.algo == "NN"):
            self.ai = nnr( par1 )
            self.ai.fit(self.dataset.insample_data,self.dataset.insample_res)

    def test(self):
        self.test_res = pd.DataFrame(self.ai.predict(self.dataset.learn_data), columns=["Res"],index=self.dataset.learn_data.index)
        self.test_cor = np.correlate(self.test_res["Res"], self.dataset.learn_res["Res"])

    def predict(self):
        pres = self.ai.predict(self.dataset.outofsample_data)
        self.res = pd.DataFrame( pres, columns=["Res"],  index = self.dataset.outofsample_data.index )
        return self.res

    def plotprediction(self):
        title = self.algo +  " " + str(self.par1) + " " + str(self.par2) + " " + " ".join( self.dataset.getnames() )
        plotequity( self.res, self.dataset.outofsample_res , title)

    def plottestandprediction(self):
        title = self.algo + " " + str(self.par1) + " " + str(self.par2) + " " + " ".join(self.dataset.getnames())
        plotequity(self.test_res.append( self.res ) , self.dataset.learn_res.append( self.dataset.outofsample_res), title)


mydata = pd.read_csv( "data/cme.csv", index_col="Date" );

ds = dataset()
ds.createfromohlcv(mydata)
ds.splitdata(0.5,0.45,0.05)
ds.normalize()
'''
ds2 = dataset()
ds2.selectedcolumns(ds, [0,1,3] )

print(ds2.getnames())

print( str(ds.dataset_width)+" "+str(ds.dataset_length))

cai = ai(ds2, "NN", 5, 0)
cai.test();
print("Correlation :", cai.test_cor)
pres = cai.predict()
cai.plottestandprediction()
'''

DATALEN = 50

x = tf.placeholder(tf.float32, [DATALEN, 4])
#y = tf.placeholder(tf.float32, [None, 1])
y_ = tf.placeholder(tf.float32, [DATALEN,1])

#W = tf.Variable(tf.zeros([4, 4]))

W = tf.Variable(tf.truncated_normal([4, 4],stddev=1.0 / 2 ),name='weights')
b = tf.Variable(tf.zeros([4]))
h1 = tf.Variable(tf.zeros([1,4]))
h2 = tf.Variable(tf.zeros([1,4]))


#y = tf.nn.softmax(tf.matmul(x, W) +  b)
h1 = tf.nn.softsign(tf.matmul(x, W) + b)
#h1 = tf.matmul(x, W) + b
h1=tf.Print(h1, [W, b], "h1 :")
h2 = tf.nn.softsign(tf.matmul(h1, W) + b)
#y = tf.matmul(h2, W) + b
y = tf.reduce_max(h2,1)
y = tf.Print( y, [h2], "Y : ")

#cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.abs(y_-y), reduction_indices=[1]))
#cross_entropy = tf.abs(y - y_)
#cross_entropy = tf.reduce_mean(tf.abs(y - y_))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]))
#cross_entropy = tf.reduce_mean(tf.abs(y - y_))
#cross_entropy =  tf.reduce_sum( tf.mul( tf.mul( tf.div( y_ ,tf.abs(y_)) , tf.div( y ,tf.abs(y)) ),y_  ) , 0 )
cross_entropy =  tf.reduce_sum(  -tf.mul(y_,y ) , 0 )
cross_entropy = tf.Print(cross_entropy,[y_], "CE :")

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

#y = tf.Print(y, [h1])

sess = tf.Session()

sess.run(init)


sess.run(train_step, feed_dict={x: ds.insample_data[0:DATALEN], y_: ds.insample_res[0:DATALEN]})



#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
correct_prediction =  tf.sub(y,0)
#accuracy = tf.cast(correct_prediction, tf.float32)
#accuracy = tf.Print(accuracy, [x,y_], message="Acc X:")
#accuracy = tf.reduce_sum(correct_prediction, reduction_indices=[0])
accuracy = correct_prediction

pres =  sess.run(accuracy, feed_dict={x: ds.outofsample_data[0:DATALEN], y_: ds.outofsample_res[0:DATALEN]})

pres = pd.DataFrame( pres , index=ds.outofsample_res.index[0:DATALEN])
print(pres)
plotequity( pres, ds.outofsample_res[0:DATALEN], "Tensorflow" )
print(sess.run(W))



#pres_learn =  sess.run(accuracy, feed_dict={x: ds.learn_data, y_: ds.learn_res})
#pres_learn = pd.DataFrame( pres_learn , index=ds.learn_res.index)

#plotequity( pres_learn.append(pres), ds.learn_res.append( ds.outofsample_res), "Tensorflow" )


sess.close()


#print(ds.dataset.iloc[:,0:ds.dataset_width-1])

#for i in pres:
#print( ds.dataset["Res"].iloc[ds.insamplelearn].iloc[pres].mean()  )

#plt.hist(pres,50, normed=0, alpha=0.75)
#plt.show()
#plt.plot(pres)
#plt.show()


#print(mydata.shape[0])
#print(mydata[1:5].shift(-1) )

#mydata2 = pd.DataFrame(mydata['Close'],index= mydata.index)
#mydata2['Res'] = mydata['Close'].shift(-1)
#mydata2['MA'] = ta.MA( mydata,10 )
#mydata2=mydata2.dropna(axis=0)



#print(mydata2[1:5])

#print(mydata.loc["2002-12-12":"2003-01-01"])

#print(mydata.loc["2011-01-01,2012-01-01"]["Open"][0:5:1] )

