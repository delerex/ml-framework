import quandl as q
import pandas as pd
import numpy as np
import config

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import RadiusNeighborsClassifier

import TA as ta

q.ApiConfig.api_key = config.quandl_key


class dataclass:
    i=0
    dataset_width = 0
    dataset_length = 0
    datares_width = 0

    dataset = pd.DataFrame()
    datares = pd.DataFrame()

    def __init__(self, df=None,  columns=None, rescolumns=None):
        i = 1
        if df != None:
            if columns == None:
                columns = range(0, df.dataset_width)
            if rescolumns == None:
                rescolumns = range(0, df.datares_width)
            self.dataset = pd.DataFrame(df.dataset.ix[:,columns])
            self.datares = pd.DataFrame(df.datares.ix[:,rescolumns])

            self.dataset_length = self.dataset.shape[0]
            self.dataset_width = self.dataset.shape[1]
            self.datares_width = self.datares.shape[1]

    def create(self, filename, rescolumns = 1, dropna=False, fillna=False):
        dataset = pd.read_csv(filename, index_col='DateTime', parse_dates=True)

        if dropna == True:
            dataset = dataset.dropna(axis=0)
        if fillna == True:
            dataset = dataset.fillna(0)

        self.dataset = pd.DataFrame(dataset.ix[:, range(0, dataset.shape[1] - rescolumns)])
        self.datares = pd.DataFrame(dataset.ix[:, range(dataset.shape[1] - rescolumns, dataset.shape[1])])

        self.dataset_length = self.dataset.shape[0]
        self.dataset_width = self.dataset.shape[1]
        self.datares_width = self.datares.shape[1]

    def prenormalize(self, sdev=0, NormRes=False):
        if sdev == 0:
            return
        for i in range(0, self.insample_data.shape[1]):
            mean = self.insample_data.iloc[:,i].mean()
            stddev = self.insample_data.iloc[:,i].std()
            self.insample_res = self.insample_res[ abs((self.insample_data.iloc[:, i] - mean)/stddev) < sdev  ]
            self.insample_data = self.insample_data[ abs((self.insample_data.iloc[:, i] - mean)/stddev) < sdev  ]
        if NormRes == True:
            for i in range(0, self.insample_res.shape[1]):
                mean = self.insample_res.iloc[:, i].mean()
                stddev = self.insample_res.iloc[:, i].std()
                self.insample_data = self.insample_data[abs((self.insample_res.iloc[:, i] - mean) / stddev) < sdev ]
                self.insample_res = self.insample_res[abs((self.insample_res.iloc[:, i] - mean) / stddev) < sdev ]


    def normalize(self, NormRes=False):
        for i in range(0, self.insample_data.shape[1]):
            mean = self.insample_data.iloc[:,i].mean()
            stddev = self.insample_data.iloc[:,i].std()
            self.insample_data.iloc[:, i] = (self.insample_data.iloc[:, i] - mean) / stddev
            if self.learn_data.shape[0] > 0: self.learn_data.iloc[:, i] = (self.learn_data.iloc[:, i] - mean) / stddev
            self.outofsample_data.iloc[:, i] = (self.outofsample_data.iloc[:, i] - mean) / stddev
        if NormRes == True:
            for i in range(0, self.insample_res.shape[1]):
                mean = self.insample_res.iloc[:, i].mean()
                stddev = self.insample_res.iloc[:, i].std()
                self.insample_res.iloc[:, i] = (self.insample_res.iloc[:, i] - mean) / stddev
                if self.learn_res.shape[0] > 0 : self.learn_res.iloc[:, i] = (self.learn_res.iloc[:, i] - mean) / stddev
                self.outofsample_res.iloc[:, i] = (self.outofsample_res.iloc[:, i] - mean) / stddev


    def postnormalize(self, sdev=0):
        return

    def getnames(self):
        return self.insample_data.columns.values

    def classify(self, sdev=0.5):
        for i in range(0, self.insample_res.shape[1]):
            self.insample_res.ix[self.insample_res.iloc[:,i] >= sdev, i] = 1
            self.insample_res.ix[self.insample_res.iloc[:, i] <= -sdev, i] = -1
            self.insample_res.ix[((self.insample_res.iloc[:, i] > -sdev) & (self.insample_res.iloc[:, i] < sdev)), i] = 0


    def splitdata_pct(self, skip_size, insample_size, test_size, outofsample_size ):
        self.insamplelearn_items = np.zeros(self.dataset_length, dtype=bool)
        self.insampletest_items = np.zeros(self.dataset_length, dtype=bool)
        self.outofsample_items = np.zeros(self.dataset_length, dtype=bool)

        self.insamplelearn_items[int(self.dataset_length *skip_size):int(self.dataset_length *skip_size + self.dataset_length*insample_size)]=True
        self.insampletest_items[int(self.dataset_length *skip_size + self.dataset_length * insample_size): int(self.dataset_length *skip_size + self.dataset_length * insample_size + self.dataset_length * test_size)] = True
        self.outofsample_items[int(self.dataset_length *skip_size + self.dataset_length * insample_size+ self.dataset_length * test_size):int(self.dataset_length *skip_size + self.dataset_length * insample_size + self.dataset_length * test_size +  + self.dataset_length * outofsample_size)] = True

        self.insample_data = self.dataset.iloc[self.insamplelearn_items, :self.dataset_width]
        self.insample_res = self.datares.iloc[self.insamplelearn_items, :self.datares_width]

        self.learn_data = self.dataset.iloc[self.insampletest_items, :self.dataset_width]
        self.learn_res = self.datares.iloc[self.insampletest_items, :self.datares_width]

        self.outofsample_data = self.dataset.iloc[self.outofsample_items, :self.dataset_width]
        self.outofsample_res = self.datares.iloc[self.outofsample_items, :self.datares_width]

    def selectedcolumns(self, dataset, columns):
        self.insample_data = dataset.insample_data.iloc[:,columns]
        self.learn_data = dataset.learn_data.iloc[:,columns]
        self.outofsample_data = dataset.outofsample_data.iloc[:, columns]
        self.insample_res = dataset.insample_res
        self.learn_res = dataset.learn_res
        self.outofsample_res = dataset.outofsample_res

    def plothist(self, datatype="is"):
        mydpi=96
        dt = self.insample_data
        dr = self.insample_res

        if datatype == "ls":
            dt = self.learn_data
            dr = self.learn_res
        if datatype == "os":
            dt = self.outofsample_data
            dr = self.outofsample_res


        colcount = 3
        rowcount = len( self.getnames()) / 3 +1
        if len( self.getnames()) % 3 != 0:
            rowcount = rowcount+1
        curplot=1
        fig = plt.figure(figsize=(1800/mydpi, 1000/mydpi), dpi=mydpi)

        for curcol in self.getnames():
            plt.subplot(rowcount,colcount, curplot)
            plt.hist(np.asarray(dt.loc[:,curcol]), bins=60)#
            plt.title(curcol)
            curplot=curplot+1

        plt.subplot(rowcount, colcount, curplot)
        plt.hist(np.asarray(dr), bins=60)
        plt.title("Res")
        plt.show()

    def plotpoints(self, col1=0, col2=1, datatype="is"):
        mydpi=96
        dt = self.insample_data
        dr = self.insample_res

        if datatype == "ls":
            dt = self.learn_data
            dr = self.learn_res
        if datatype == "os":
            dt = self.outofsample_data
            dr = self.outofsample_res
        fig = plt.figure(figsize=(1000 / mydpi, 1000 / mydpi), dpi=mydpi)
 #       plt.ion()
        ax = fig.add_subplot(111)
        x=np.asarray(dt.iloc[:,col1])
        y=np.asarray(dt.iloc[:,col2])
        r =np.asarray(dr.iloc[:,0])
        ax.scatter(x[r>0], y[r>0], s=r*50, c="g" )
        ax.scatter(x[r<=0], y[r<=0], s=r*50, c="r" )
        plt.show()
#        plt.pause(0.001)



    def nnsmooth(self, columns=None, rescolumn=None, k=10, cycles=1):
        if columns == None:
            columns = range(0, self.dataset_width )
        colcnt = len(columns)
        dt = self.insample_data
        dataset = pd.DataFrame(dt.ix[:,columns])
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(dataset)
        distabnce, indicies = nbrs.kneighbors(dataset)

        for i in range(0, cycles):
            dr = self.insample_res
            for x in indicies:
                mn= self.insample_res.ix[  x[range(1,k)], 0 ].mean()
                dr.ix[x[0],0] = dr.ix[x[0],0] * 0.8 + mn * 0.2
                self.insample_res = dr

    def nnradiussmooth(self, columns=None, rescolumn=None, distance=0.2, cycles=1):
        if columns == None:
            columns = range(0, self.dataset_width)
        colcnt = len(columns)
        dt = self.insample_data
        dataset = pd.DataFrame(dt.ix[:, columns])
        nbrs = RadiusNeighborsClassifier().fit(dt, np.zeros_like(self.insample_res).reshape(self.insample_res.shape[0],))
        nb  = nbrs.radius_neighbors(dt, distance, return_distance=False )

        for i in range(0, cycles):
            dr = self.insample_res
            for x in nb:
                mn = self.insample_res.ix[x, 0].mean()
                dr.ix[x[0], 0] = dr.ix[x[0], 0] * 0.8 + mn * 0.2
                self.insample_res = dr

    def nncut(self, distance=1, type='inner', datatype='all'):
        def nncut_proc( distance, dt, dr, type):
            if dt.shape[0] == 0:
                return [dt,dr]
            nbrs = RadiusNeighborsClassifier().fit(dt, np.zeros_like(dr).reshape(dt.shape[0],))
            colcnt = dt.shape[1]
            middle = nbrs.radius_neighbors(np.zeros(colcnt).reshape(1,colcnt), distance, return_distance=False)
            if type == 'inner':
                dt = dt.drop(dt.index[np.asarray(middle[0])])
                dr = dr.drop(dr.index[np.asarray(middle[0])])
            if type == 'outer':
                dt = dt[ dt.index.isin( dt.index[ np.asarray(middle[0]) ] )]
                dr = dr[ dr.index.isin( dr.index[ np.asarray(middle[0]) ] )]
            return [dt, dr]

        if datatype == 'is':
            self.insample_data, self.insample_res = nncut_proc(distance, self.insample_data, self.insample_res, type)
        if datatype == 'os':
            self.outofsample_data, self.outofsample_res = nncut_proc(distance, self.outofsample_data, self.outofsample_res,type)
        if datatype == 'all':
            self.insample_data, self.insample_res = nncut_proc(distance, self.insample_data, self.insample_res, type)
            self.learn_data, self.learn_res = nncut_proc(distance, self.learn_data, self.learn_res, type)
            self.outofsample_data, self.outofsample_res = nncut_proc(distance, self.outofsample_data, self.outofsample_res,type)

    def errorfunc(self, res, pred, verbose = False ):
        result = pd.DataFrame( res )
        result["Pred"] = pred
        result_series =  pd.Series( [tuple(i) for i in result.values] )

        rc = result_series.value_counts()
        positive= 0
        negative = 0
        falsepositive=0
        falsenegative = 0

        if (1,1) in rc.index:
            positive = rc[(1,1)]
        if (0,0) in rc.index:
            negative = rc[(0,0)]
        if (0,1) in rc.index:
            falsepositive = rc[(0,1)]
        if (1,0) in rc.index:
            falsenegative = rc[(1,0)]

        ret = - ( (falsenegative * 100.0 / (positive + falsenegative) * 1) + (falsepositive * 100.0 / (negative + falsepositive) * 2 ) )

        if( verbose == True ):
            print( "Positive rate {0:0.2f}%.  Negative rate {1:0.2f}%. Score : {2:0.2f}".format(  positive * 100.0 / (positive + falsenegative) , negative * 100.0 / (negative + falsepositive)  , ret ) )
        return ret

    def filter(self, sdev=3, printerror = False):
        if sdev == 0:
            return
        print(self.insample_data.shape[1])
        res = np.ones(self.outofsample_res.shape[0], dtype=np.int8)
        for i in range(0, self.insample_data.shape[1]):
            mean = self.insample_data.iloc[:,i].mean()
            stddev = self.insample_data.iloc[:,i].std()
            tmp = np.where(abs((self.outofsample_data.iloc[:, i] - mean) / stddev) < sdev, 1, 0)
            res = np.bitwise_and(res, tmp)

            if printerror == True:
                print (  self.outofsample_data.columns.values[i] )

                print("Filtered {0} items {1:0.2f}%".format( len(tmp) - np.sum(tmp),  (len(tmp) - sum(tmp)) * 100.0 / len(tmp) ) )
                self.errorfunc( self.outofsample_res[[0]], tmp, True)


        print("After filtering")

        if printerror == True:
            self.errorfunc( self.outofsample_res[[0]], res, True )

        befores = self.outofsample_res.shape[0]

        self.outofsample_res = self.outofsample_res[ res == 1 ]
        self.outofsample_data = self.outofsample_data[ res == 1 ]
        print("Filtered {0} items {1:0.2f}%".format( befores - self.outofsample_res.shape[0], ((befores - self.outofsample_res.shape[0]) * 100.0 / befores)   ))
        return res


    def save(self, filename="data/dataset.txt", dropna=False, fillna=False):
        ds = self.dataset
        ds = ds.join(self.datares)
        if dropna == True:
            ds = ds.dropna(axis=0)
        if fillna == True:
            ds = ds.fillna(0)

        ds.to_csv(filename, float_format="%.4f")



class dataclass_ohlcv(dataclass):
    def create(self, ohlcv):
        dataset = pd.DataFrame(ohlcv['Close'].diff(), columns=['Close'], index=ohlcv.index)
        dataset['MA'] = ( ta.MA(ohlcv, 10) - ohlcv['Close']) / ( ta.ATR(ohlcv, 10))
        dataset['MA2'] = ( ta.MA(ohlcv, 20) - ohlcv['Close']) / ( ta.ATR(ohlcv, 10))
        dataset['Res'] = ohlcv['Close'].shift(-1).diff()
        dataset = dataset.dropna(axis=0)

        self.dataset = pd.DataFrame(dataset.ix[:, range(0, dataset.shape[1] - 2)])
        self.datares = pd.DataFrame(dataset.ix[:, ((dataset.shape[1] - 1))])

        self.dataset_length = self.dataset.shape[0]
        self.dataset_width = self.dataset.shape[1]
        self.datares_width = self.datares.shape[1]



class dataclass_syseval(dataclass):
    def create(self, filename):
        dataset = pd.read_csv(filename, index_col = 'DateTime')

        dataset = dataset[ dataset['LongSD'] != 0 ]
        dataset = dataset[ dataset['ShortSD'] != 0 ]

        self.dataset = pd.DataFrame( dataset.ix[:, range(0, dataset.shape[1] - 6)] )
        self.datares = pd.DataFrame( dataset.ix[:, range( dataset.shape[1] -6 , dataset.shape[1]  )])



        self.dataset_length = self.dataset.shape[0]
        self.dataset_width = self.dataset.shape[1]
        self.datares_width = self.datares.shape[1]


    def classify(self, n=20):
        for i in range(0, self.insample_res.shape[1]):
            #print(self.insample_res.iloc[:, i])
            #np.which( self.insample_res.iloc[:, i] >= n, 0, 1 )
            self.insample_res.ix[self.insample_res.iloc[:, i] < n, i] = 1
            self.insample_res.ix[self.insample_res.iloc[:, i] >= n, i] = 0


            self.outofsample_res.ix[self.outofsample_res.iloc[:, i] < n, i] = 1
            self.outofsample_res.ix[self.outofsample_res.iloc[:, i] >= n, i] = 0


class dataclass_syseval_binary(dataclass):
    def create(self, filename, dropna=False, fillna = False):
        dataset = pd.read_csv(filename, index_col = 'DateTime', parse_dates=True )

        if dropna == True:
            dataset = dataset.dropna(axis=0)
        if fillna == True:
            dataset = dataset.fillna(0)

        self.dataset = pd.DataFrame( dataset.ix[:, range(0, dataset.shape[1] - 6)] )
        self.datares = pd.DataFrame( dataset.ix[:, range( dataset.shape[1] -6 , dataset.shape[1]  )])

        self.dataset_length = self.dataset.shape[0]
        self.dataset_width = self.dataset.shape[1]
        self.datares_width = self.datares.shape[1]


    def classify(self, n=20):
        for i in range(0, self.insample_res.shape[1]):
            self.insample_res.ix[self.insample_res.iloc[:, i] < n, i] = 1
            self.insample_res.ix[self.insample_res.iloc[:, i] >= n, i] = 0


            self.outofsample_res.ix[self.outofsample_res.iloc[:, i] < n, i] = 1
            self.outofsample_res.ix[self.outofsample_res.iloc[:, i] >= n, i] = 0

    def binaryclassify(self, column=0, n=0 , reverse=False):

            if reverse == True:
                greater = 0
                lower = 1
            else:
                greater = 1
                lower = 0

            self.insample_res[[column]] = np.where( self.insample_res[[column]] > n, greater, lower )
            self.outofsample_res[[column]] = np.where( self.outofsample_res[[column]] > n, greater, lower )



