'''
Created on Apr 15, 2021
Alex
'''

import pandas as pd
import numpy as np
import os.path
import math
import time
import websocket
import json
from binance.client import Client
import csv
import talib as ta
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.misc import electrocardiogram

binance_api_key = ''
binance_api_secret = ''

client = Client(api_key=binance_api_key, api_secret=binance_api_secret)

socket = 'wss://stream.binance.com:9443/ws/btcusdt@kline_1m'

macd, signal, hist, pp, R1, S1 = [], [], [], [], [], []
btc_df = pd.read_csv('C:\\Users\\Alex\\Downloads\\btc_test.csv')
btc_df.set_index('date', inplace=True)

'''def btc_historical():
    global btc_df
    print("Rewriting csv")
    bars = client.get_historical_klines('BTCUSDT', '4h', start_str='1200 hours ago UTC', limit=500)

    for line in bars:
        del line[6:]

    btc_df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close', 'vol'])
    btc_df.set_index('date', inplace=True)
    btc_df.to_csv('C:\\Users\\Alex\\Downloads\\btc_test.csv')

if not os.path.isfile('C:\\Users\\Alex\\Downloads\\btc_test.csv'):
    btc_historical()
    

else:
    btc_df = pd.read_csv('C:\\Users\\Alex\\Downloads\\btc_test.csv')
    btc_df.set_index('date', inplace=True)

    last_Time = client.get_historical_klines('BTCUSDT', '1m', start_str='1 min ago UTC')
    last_Time = next(zip(*last_Time))
    last_Time = ''.join([str(i) for i in last_Time])

    if last_Time != btc_df.tail(1).index[-1]:
        btc_historical()
'''


def get_MACD():
    global macd, signal, hist
    macd, signal, hist = ta.MACD(btc_df['close'])
    macd = macd.tolist()
    signal = signal.tolist()
    hist = hist.tolist()


# find way to resample data so it calculates pivots only a day at a time, get rid of ms in index
# could use support levels as a place to sell to limit loss or resistance
# check out how we find up and down trends and see if it can be optomized
# derivative of tan is 0 when there is a peak or valley???
# https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/43512887#43512887


def pivots():
    global pp, R1, S1
    pp = ((btc_df['high'] + btc_df['low'] + btc_df['close']) / 3)
    R1 = (2 * pp - btc_df['low'])
    S1 = (2 * pp - btc_df['high'])
    pp = pp.tolist()
    R1 = R1.tolist()
    S1 = S1.tolist()


def order_type():
    global btc_df, macd, signal, hist, pp, R1, S1

    w, q, x, y, l = 0, 0, 0, 0, 0
    flag = True
    total = 10000
    buy, sell = [], []
    overB = 0
    nobuytoday = 0

    pivots = []
    counter = 0
    lastPivot = 0
    Range = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    while x < len(btc_df.index):

        currentMax = max(Range, default=0)
        value = round(btc_df.iloc[x]['high'], 2)
        Range = Range[1:9]
        Range.append(value)

        if currentMax == max(Range, default=0):
            counter += 1
        else:
            counter = 0
        if counter == 6:
            lastPivot = currentMax
            pivots.append(lastPivot)

        if pivots:
            if q < (len(pivots) - 1) and len(pivots) >= 2:
                if pivots[q] > pivots[q + 1] and q < (len(pivots) - 1):
                    print('down trend')
                    flag = False
                    q += 1

                elif pivots[q] < pivots[q + 1] and q < (len(pivots) - 1):
                    print('up trend')
                    flag = True
                    q += 1

        if nobuytoday == 4:
            nobuytoday = 0

        if macd[x] > signal[x] and len(
                buy) < 6 and flag == True and nobuytoday == 0:  # and btc_df.iloc[x]['open'] > R1[x+1]:
            buy.append(btc_df.iloc[x]['close'])
            print('bought at ' + str(btc_df.iloc[x]['close']))
            total -= int(float((btc_df.iloc[x]['close']) / 18))
            l += 1

        while buy != [] and y < len(buy):
            if (buy[y] * 1.01) < btc_df.iloc[x]['close']:
                sell.append(btc_df.iloc[x]['close'])
                total += int(float((btc_df.iloc[x]['close']) / 18))
                print('Profit taken  at ' + str(btc_df.iloc[x]['close']))
                del buy[y]
                w += 1
                y += 1
                continue

            if signal[x] > macd[x] and (buy[y] * .95) > btc_df.iloc[x]['close']:  # > S1[x + 1]:
                sell.append(btc_df.iloc[x]['close'])
                total += int(float((btc_df.iloc[x]['close']) / 18))
                print('Stop Loss at ' + str(btc_df.iloc[x]['close']))
                del buy[y]
                y += 1
                continue

            if y == len(buy) - 1:
                y = 0
                break

            '''if not flag:
                while overB < len(buy):
                    print('Closed Out')
                    sell.append(buy[overB])
                    total += int(float((btc_df.iloc[x]['close']) / 18))
                    del buy[overB]
                    w += 1
                    overB += 1

                overB = 0
                y += 1
                continue'''

            y += 1

        x += 1
        nobuytoday += 1
        print(total)


# pivots()
# get_MACD()
# order_type()


def PriceAction(a):
    global btc_df
    btc_df = btc_df.reset_index()

    x = btc_df.index.to_numpy()

    peaks = find_peaks(btc_df['high'])
    plt.plot(btc_df['high'])

    plt.show()

u = 0
PriceAction(u)

'''if u == 0:
    u = 6
    PriceAction(u)

if u == 6:
    u = 120
    PriceAction(u)
'''

'''print('starting')
def on_message(ws, message):
    global btc_df
    json_message = json.loads(message)
    print(json_message)
    candle = json_message['k']
    is_candle_closed = candle['x']
    date = candle['t']
    open = candle['o']
    high = candle['h']
    low = candle['l']
    close = candle['c']
    vol = candle['v']

    if is_candle_closed:
        dic = {'date': [date], 'open': [open], 'high': [high], 'low': [low], 'close': [close], 'vol': [vol]}
        new_df = pd.DataFrame.from_dict(dic, orient='columns')
        new_df.set_index('date', inplace=True)
        btc_df = btc_df.append(new_df)




        # 1) Compare 1 day HnL with 4h HnL
        # 2) Set Global vars for macd, btc_df everything...
        # 3) Be able to execute trades
        # 4) Be able to reconnect if issues arise


def on_close(ws):
    print('### Connection Closed ###')

ws = websocket.WebSocketApp(socket,  on_message=on_message,  on_close=on_close)
ws.run_forever()
'''
'''
timestamp = client._get_earliest_valid_timestamp('BTCUSDT', '4h')
bars = client.get_historical_klines('BTCUSDT', '4h', timestamp, limit=1000)

# delete unwanted data - just keep date, open, high, low, close
for line in bars:
    del line[5:]

# option 4 - create a Pandas DataFrame and export to CSV
btc_df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close'])
btc_df.set_index('date', inplace=True)

# export DataFrame to csv
btc_df.to_csv('btc_bars2.csv')
'''
