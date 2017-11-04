import pymongo
import pandas as pd
import numpy as np
client_mongo = pymongo.MongoClient()
db_mongo = client_mongo.datafeed
from datetime import datetime, date
from decimal import *


datetimeformat = "%Y-%m-%d"
getcontext().prec = 6

coindata={
    "USD": {
        "name": "US Dollar",
        "shortname": "USD",
        "magnitude": 100,
        "rounding": 2

    },
    "BTC": {
        "name": "Bitcoin",
        "shortname": "BTC",
        "magnitude": 100000000,
        "rounding": 6

    },
    "ETH": {
        "name": "Ether",
        "shortname": "ETH",
#        "magnitude": 1000000000000000000,
         "magnitude": 10000000000000000,
        "rounding": 6

    },
    "ETC": {
        "name": "Ethereum Classic",
        "shortname": "ETC",
        "magnitude": 10000000,
        "rounding": 6

    },
    "TNT": {
        "name": "Tierion",
        "shortname": "TNT",
        "magnitude": 100000000,
        "nomination" : "ETH",
        "startdate" : "2017-08-25",
        "rounding": 6,
        "type" : "ERC20",
        "contract" : "0x08f5a9235b08173b7569f83645d2c7fb55e8ccd8"
    },
    "BTCToken": {
        "name": "Demo Fund BTC Token",
        "shortname": "BTCToken",
        "magnitude": 10000000,
        "rounding": 6
    },
    "common" :{
        "name": "common",
        "shortname": "common",
        "magnitude": 10000000,
        "rounding" : 6
    }

}


def FloatToInt(amount, currency):
    if currency not in coindata:
        return int(float(amount) * float(coindata["common"]["magnitude"]))
    return int( float(amount) * float(coindata[currency]["magnitude"]))


def IntToFloat(amount, currency):
    if currency not in coindata:
        return round( float( float( amount ) / float(coindata["common"]["magnitude"])), coindata["common"]["rounding"] )
    return round( float( amount ) / float(coindata[currency]["magnitude"]), coindata[currency]["rounding"] )

def get_coindata(currency):
    if currency in coindata:
        return coindata[currency]
    return{
        "name": currency,
        "shortname": currency,
        "magnitude": 10000000,
        "rounding" : 6
    }


def get_rates(currency, basiccurrency, datetime_from=None, datetime_to=None):
    r = db_mongo[currency + "_" + basiccurrency].find({}).sort("Time", 1)
    df = pd.DataFrame.from_records(r)
    if df.empty:
        return None
    s = pd.Series( np.asarray(df["Close"]), index = pd.to_datetime(df["Time"], unit="s") )
    s.name = currency
    return s[datetime_from : datetime_to]

def get_rate(currency, basiccurrency, time=None ):
    if currency == basiccurrency:
        return 1
    if isinstance(time, basestring):
        time = datetime.strptime(time, datetimeformat)
    if isinstance(time, datetime):
        time = int( datetime.strftime(time, "%s"))
    if isinstance(time, date):
        time = int( datetime.strftime(time, "%s"))

    startdate = None
    if time is not None:
        if "startdate" in get_coindata(currency):
            startdate = int( datetime.strptime(get_coindata(currency)["startdate"], datetimeformat).strftime("%s"))

        r = db_mongo[currency + "_" + basiccurrency].find({"Time": {"$lte": int(time)}}).sort("Time", -1).limit(1)
        if r.count() == 0 or (startdate is not None and startdate > time):
            if currency in coindata and "nomination" in get_coindata(currency):
                return get_rate(coindata[currency]["nomination"], basiccurrency, time)
            return 0
        rt = r.next()
        return rt["Close"]
    r = db_mongo[currency + "_" + basiccurrency].find({}).sort("Time", -1).limit(1)
    if r.count() == 0:
        return 0
    return r.next()["Close"]

def get_time(currency, basiccurrency, time=None):
    if currency == basiccurrency:
        return None

    if time is None:
        r = db_mongo[currency + "_" + basiccurrency].find({}).sort("Time", -1).limit(1)
        if r.count() == 0:
            return None
        return int(r.next()["Time"])
    else:
        if isinstance(time, basestring):
            time = datetime.strptime(time, "%Y-%m-%d")
        if isinstance(time, int):
            time = datetime.fromtimestamp(time)
        r  = db_mongo[currency + "_" + basiccurrency].find_one({"_id":time.strftime("%Y-%m-%d")})
        if r is not None:
            return int(r["Time"])

    return None


def drop_rates(currency, basiccurrency):
    db_mongo[currency + "_" + basiccurrency].drop()


def set_rates(currency, basiccurrency, series ):



    for i in series.index:
        r = db_mongo[currency + "_" + basiccurrency].find_one({"_id":datetime.strftime(i, datetimeformat)})
        if r is not None:
            r["Close"]=series[i]
            db_mongo[currency + "_" + basiccurrency].save(r)
        else:
            db_mongo[currency + "_" + basiccurrency].insert({"_id": datetime.strftime(i, datetimeformat), "Close":series[i], "Time" : int( datetime.strftime(i, "%s")) })



def ohlcv(currency, basiccurrency, datetime = None):
    if datetime is not None:
        r = db_mongo[currency + "_" + basiccurrency].find( {"_id": datetime} )
    else:
        r = db_mongo[currency + "_" + basiccurrency].find({}).sort("Time", -1).limit(1)
        if r.count() == 0:
            return None
        r = r.next()
    return r


def close(currency, basiccurrency, time = None):
    ohlc = ohlcv(currency,basiccurrency, time)
    if "Close" not in ohlc:
        return None
    return ohlc["Close"]


def volume_from( currency, basiccurrency, time = None):
    ohlc = ohlcv(currency,basiccurrency, time)
    if "VolumeFrom" not in ohlc:
        return None
    return ohlc["VolumeFrom"]


def volume_to(currency, basiccurrency, time = None):
    ohlc = ohlcv(currency,basiccurrency, time)
    if "VolumeTo" not in ohlc:
        return None
    return ohlc["VolumeTo"]


def convert(currency, basiccurrency, amount, time = None):
    rate = close(currency, basiccurrency, time)
    return rate * amount


def get_closes(watchlist, basiccurrency, time_from=None, time_to=None):
    df = None
    for s in watchlist:
        if s is not basiccurrency:
            r = get_rates(s, basiccurrency, time_from, time_to)
            if df is None:
                df = pd.DataFrame(r)
            else:
                df = df.join(r)
    return df


