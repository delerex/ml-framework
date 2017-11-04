import pymongo
import requests
import json
import pandas as pd
import numpy as np
import currencies
import requests
import time
from datetime import datetime



client = pymongo.MongoClient()
db = client.datafeed

def getprice(symbol, tosymbols):
    try:
        print( "GetDayData " + symbol + " " + tosymbols)
        r = requests.get("https://min-api.cryptocompare.com/data/price",
                         params={'fsym': symbol, 'tsyms': tosymbols})
        rj = r.json()
        return rj
    except:
        return None



def getdaydata(symbol, tosymbols, limit = 200):
    try:
        print( "GetDayData " + symbol + " " + tosymbols)
        r = requests.get("https://min-api.cryptocompare.com/data/histoday",
                         params={'fsym': symbol, 'tsym': tosymbols, 'limit': limit, 'aggreagte': 1, 'e': 'CCCAGG', 'alldata': 'true' })
        rj = r.json()
        ds = pd.DataFrame(rj["Data"])
        ret = pd.DataFrame( {"Time" : np.asarray(ds['time']), "Open" : np.asarray(ds['open']), "High" : np.asarray(ds['high']), "Low": np.asarray(ds['low']),
                            "Close": np.asarray(ds['close']), "VolumeFrom": np.asarray(ds['volumefrom']), "VolumeTo": np.asarray(ds['volumeto'])},
                            index=pd.to_datetime(ds['time'], unit="s" ).apply( lambda x :  x.strftime("%Y-%m-%d"))  ).to_dict("index")

        return ds
    except:
        return None


def loaddata(symbols, basicsymbol):
    for t in symbols:
        db.drop_collection(t+"_"+basicsymbol)
        if t is not basicsymbol:
            time.sleep(3)
            cl = None
            while cl is None:
                cl = getdaydata(t, basicsymbol)
            for i in cl:
                item = cl[i]
                item["_id"] = i
                db[t + "_" +basicsymbol].insert( item )



def updatedata(symbols, basicsymbol, days):
    for t in symbols:
        if t is not basicsymbol:
            time.sleep(3)
            if days is None:
                lt = currencies.get_time(t, basicsymbol)
                print datetime.fromtimestamp(lt)
                days_limit = (datetime.now() - datetime.fromtimestamp(lt) ).days
            else:
                days_limit = days
            print days_limit
            cl = getdaydata(t, basicsymbol, days_limit)
            print cl
            for i in cl:
                item = cl[i]
                ci = currencies.get_time(t, basicsymbol, i)
                if ci is not None:
                    if item["Close"] != 0:
                        item["_id"] = i
                        r = db[t + "_" + basicsymbol].save(item)
                        print "Saved " + r
                    else:
                        print "skiped"
                else:
                    item["_id"] = i
                    r = db[t + "_" +basicsymbol].insert( item )
                    print "Inserted " + r




def buildclosedataframe(symbols, basicsymbol ):

    df = pd.DataFrame.from_records( db["dailydata_" +basicsymbol].find_one({"_id":basicsymbol})['data'])
    res = pd.DataFrame( np.asarray(df['Close']) , index = pd.to_datetime(df.index), columns=[basicsymbol] )
    for t in symbols:
        df = pd.DataFrame.from_records( db["dailydata_" + basicsymbol].find_one({"_id": t})['data'] )
        df1 = pd.DataFrame( np.asarray(df["Close"]), index=pd.to_datetime(df.index), columns=[t])
        res = res.join( df1, how="outer")

    res.index = res.index.strftime("%Y-%m-%d")
    return res.to_dict()





def build_closes():
    db.drop_collection("frames")
    closeframe = buildclosedataframe(portfolio, "BTC")
    db.frames.insert({"_id": "Closes", "data": closeframe})


#    v.to_csv("data/result.csv", index_label="Time")


#print( pd.DataFrame.from_records(db["dailydata_" + "BTC"]["BTC"]) )

#for i in db.dailydata_BTC.find({"_id":"BTC"}):
#    print( pd.DataFrame.from_records(i['data'] ))

#df = pd.DataFrame.from_records( cl1.Data )

#print(db.dailydata.ETH['data'] )

#print(df)



#loaddata(portfolio, "BTC")
#loaddata(portfolio, "USD")



#db.dailydata.insert({"_id": "BTC", "data": getdaydata("BTC", "USD")})





client.close()
