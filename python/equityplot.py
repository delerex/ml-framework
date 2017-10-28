from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import datetime
import matplotlib.pyplot as plt
import numpy as np

def plotequity_pred(pred, res, title, dataformat="%Y-%m-%d"):
    mydpi = 96
    plt.close("all")
    pos =  np.where( pred > 0, 1, np.where( pred <-0, -1, 0 ) )
    eq = np.cumsum( pos * res )
    fig = plt.figure(figsize=(1800 / mydpi, 1000 / mydpi), dpi=mydpi)

    ax = fig.add_subplot(111)
    ax.autoscale_view()

    print(res.head())

    years = YearLocator()  # every year
    months = MonthLocator()  # every month
    yearsFmt = DateFormatter('%Y')

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)

    datemin =datetime.date(datetime.datetime.strptime(res.index.min(), dataformat ).year, 1, 1)
    datemax =datetime.date(datetime.datetime.strptime(res.index.max(), dataformat ).year+1, 1, 1)

    ax.set_xlim( datemin,  datemax)

    ax.fmt_xdata = DateFormatter('%Y-%m-%d')


    ax.plot( res.index, eq.values, color= "green"  )
    ax.plot( res.index, np.cumsum(res).values, color="black"  )
    fig.autofmt_xdate()
    plt.title(title)
    plt.show()


def plotequity_trades( trades, res = None, title="Title", dataformat="%Y-%m-%d"):
    mydpi = 96
    plt.close("all")
    eq = np.cumsum( trades )
    fig = plt.figure(figsize=(1800 / mydpi, 1000 / mydpi), dpi=mydpi)

    ax = fig.add_subplot(111)
    ax.autoscale_view()

    years = YearLocator()  # every year
    months = MonthLocator()  # every month
    yearsFmt = DateFormatter('%Y')

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)

    datemin =datetime.date(datetime.datetime.strptime(res.index.min(), dataformat ).year, 1, 1)
    datemax =datetime.date(datetime.datetime.strptime(res.index.max(), dataformat ).year+1, 1, 1)

    ax.set_xlim( datemin,  datemax)

    ax.fmt_xdata = DateFormatter('%Y-%m-%d')

    ax.plot( trades.index, eq.values, color= "green"  )
    if res is not None:
        ax.plot( res.index, np.cumsum(res).values, color="black"  )
    fig.autofmt_xdate()
    plt.title(title)
    plt.show()

def plotequity_pl( trades, title="Title" ):
    mydpi = 96
    plt.close("all")
    fig = plt.figure(figsize=(1800 / mydpi, 1000 / mydpi), dpi=mydpi)
    ax = fig.add_subplot(111)
    ax.autoscale_view()
    for x in trades:
        eq = np.cumsum( x )
        ax.plot( range(0,eq.shape[0]), eq, color= "green"  )
    plt.title(title)
    plt.show()