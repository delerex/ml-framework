from aiclass import ai
import equityplot
from dataclass import dataclass, dataclass_ohlcv
import pandas as pd
import quandl as q




mydata = q.get("BITFINEX/BTCUSD")
mydata.rename(columns={"Last":"Close"}, inplace=True)
print mydata
mydata.to_csv("data.csv")
mydata = pd.read_csv( "data.csv", index_col="Date" )


ds = dataclass_ohlcv()
ds.create(mydata)

ds2 = dataclass(ds, (0,1),(0))

ai1 = ai(ds2, "NNR", 5, 0)
reslist = []

print(ai1.dataset)

for i in range(0, 90, 2):
    print i
    ai1.dataset.splitdata_pct(i/100.0, 0.05,0.0,0.02)
    ai1.dataset.prenormalize(5)
    ai1.dataset.normalize()
#    ai1.dataset.nncut(0.5, datatype='all')
#    ai1.dataset.nncut(2.5, type='outer', datatype='all')
#    ai1.dataset.nnradiussmooth(distance=0.4, cycles=3)
#    ai1.dataset.nnsmooth( k = 10, cycles=3)
#    ai1.dataset.nncut(0.5, datatype='all')
#    ai1.dataset.nncut(2.5, type='outer', datatype='all')
#    ai1.dataset.nncut(1.0, datatype='all')
    ai1.fit()
    rs = ai1.predict()
    reslist.append(rs)

res = pd.concat(reslist)

print(res)

equityplot.plotequity_pred(pd.DataFrame(res), pd.DataFrame(ds2.datares.ix[res.index,0]), "Delerex.com ML framework", dataformat="%Y-%m-%d" )
