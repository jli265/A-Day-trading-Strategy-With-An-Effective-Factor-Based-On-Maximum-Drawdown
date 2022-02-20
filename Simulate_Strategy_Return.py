import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import calendar
import datetime
import os
import numpy as np
import time
import matplotlib

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.switch_backend('agg')
class Create_Return:
    def __init__(self):
         self.data = pd.read_csv('./data.csv').rename(columns={'Unnamed: 0': 'trade_time'})

    def single_strategy(self, obs_end_time=50,cut_loss_threshold=0.005,emotion_stationary_threshold=0.0009,transaction_cost=0.0002, leverage = 8.33):
         '''

         input 函数输入参数
         obs_end_time  int   某天上午开盘后到观察期结束的分钟数 默认50
         cut_loss_threshold float 止损阙值 默认0.005
         emotion_stationary_threshold float 情绪平稳度阙值 默认0.0009
         transaction_cost float 双边交易成本 默认0.0002
         leverage float 杠杆 默认8.33
         output 函数输出参数
         res_df dataframe columns: tradeday str 交易日, emotion_stationary float 观察期市场平稳度 , market_price_close float 沪深300日收盘价,  net_equity float 净资产, return 收益率

         '''
         # process input data
         data = pd.DataFrame({'trade_day':self.data['trade_time'].apply(lambda x: str(x)[:10]), 'trade_time': self.data['trade_time'].apply(lambda x: str(x)[11:]),
                              'open': self.data['open'] , 'close': self.data['close'] , 'high': self.data['high'], 'low': self.data['low'],
                              'daily_obs_max_drawdown_rate': [np.nan]*len(self.data['trade_time']),'net_equity': [1000000000]*len(self.data['trade_time'])
                              } )

         # 初始化结果中需要的参数
         res = [] #二位列表
         date_prev  = '0000-00-00' #初始化上一天日期
         # 计算结果
         for index, row in data.iterrows():
             # 按论文要求，跳过9:30之前的盘前和15:00之后的盘后的数据
             if (row['trade_time'][:2] == '09' and float(row['trade_time'][3:5]) < 30)  \
                     or  (row['trade_time'][:2] == '15' and float(row['trade_time'][3:5]) >= 1)    :
                 continue
             #--------------------------------------------遍历开盘到收盘------------------------------------------------
             date_curr = row['trade_day']
             if date_curr != date_prev: #9:30新的交易日开盘了！初始化各种参数
                 v,v_2 = float('-inf'),float('inf') #初始化沪深三百价格(v用作做多,v_2用作做空）
                 m,m_2 = 0,0 #初始化最大回撤为0(m用作做多,m_2用作做空）
                 lst,lst_2 = [], [] #初始化最大回撤列表（lst用作做多,lst_2用作做空）
                 open = row['open'] #当日开盘价
                 is_open = [0,None,None]  #初始化判断是否开仓(0不开，1开） , 方向（0做空，1做多）, 开仓价位
                 is_close = [0] #初始化判断日内是否平仓过至少一次（0没有，1大于一次）
                 is_obs = 1  #初始化判断是否处于观察期（0不处于，1处于）
                 net_equity = 1000000000 if res == [] else res[-1][-2] #初始化净资产
                 date_prev = date_curr
             if is_obs == 1: #判断是否在观察期，处于观察期判断回撤
               v = max(v,  row['high'])          #日内最高价（剔除未来函数）
               m = max(1 - row['low'] / v, m, 0) #日内最大回撤（做空，剔除未来函数）
               v_2 = min(v,  row['low'])  #日内最低价（剔除未来函数）
               m_2 = max(row['high']/ v_2 - 1, m_2,0) #日内最大回撤（做空，剔除未来函数）
               lst.append(m)
               lst_2.append(m_2)
             if is_obs == 1 and len(lst) == obs_end_time: #判断观察期是否到了结束时刻，结束时计算平稳度，收盘价到收盘才更新，更新别的项
                 emotion_stationary = min(sum(lst)/len(lst),sum(lst_2)/len(lst_2))
                 res.append([row['trade_day'], emotion_stationary, np.nan,net_equity,net_equity/1000000000-1])
                 is_obs = 0
             if is_obs == 0 and emotion_stationary <= emotion_stationary_threshold and is_open[0] == 0 and is_close[0] == 0: #观察期结束后在没有开仓下判断是否开仓
                 if open < row['close']:
                     is_open[0], is_open[1],is_open[2] = 1, 1, row['close']
                     net_equity +=  -transaction_cost*net_equity*leverage
                 if open > row['close']:
                     is_open[0], is_open[1],is_open[2] = 1, 0, row['close']
                     net_equity +=  -transaction_cost*net_equity*leverage
             if is_open[0] == 1: #开仓后判断是否止损或收盘平仓
                 if is_open[1] == 1 and (row['low'] - is_open[2])/is_open[2] <= -cut_loss_threshold: #做多止损平仓
                     net_equity += ( row['low']-is_open[2])/is_open[2]*leverage*net_equity -  leverage * net_equity * transaction_cost
                     res[-1][-2] = net_equity
                     is_open[0] = 0
                     is_close[0] = 1
                 if is_open[1] == 0 and (row['high'] - is_open[2])/is_open[2] >= cut_loss_threshold: #做空止损平仓
                     net_equity +=  -(row['high'] - is_open[2])/is_open[2] * leverage *net_equity -  leverage * net_equity * transaction_cost
                     res[-1][-2] = net_equity
                     is_open[0] = 0
                     is_close[0] = 1
                 if is_open[1] == 1 and row['trade_time'] == '15:00:00' : #做多收盘平仓
                     net_equity +=  ((row['close'] - is_open[2])/is_open[2]) * leverage * net_equity -  leverage * net_equity * transaction_cost
                     res[-1][-2] = net_equity
                     res[-1][-2] = net_equity
                     is_open[0] = 0
                     is_close[0] = 1
                 if is_open[1] == 0 and row['trade_time'] == '15:00:00' : #做空收盘平仓
                     net_equity +=  ((is_open[2]- row['close'])/is_open[2]) * leverage * net_equity  -  leverage * net_equity * transaction_cost
                     res[-1][-2] = net_equity
                     is_open[0] = 0
                     is_close[0] = 1
                 res[-1][-1] = res[-1][-2]/1000000000-1
             if  row['trade_time'] == '15:00:00' : #收盘时无论今日有无交易，更新结果中的收盘价
                 res[-1][2] = row['close']
         res_df = pd.DataFrame(res,columns=['trade_day','emotion_stationary','market_price_close','net_equity','return'])

         return  res_df

    def figure_9(self):
        '''
        input 函数输入参数 无
        output 函数输出参数 无
        '''
        df = self.single_strategy(obs_end_time=50, cut_loss_threshold=0.005, emotion_stationary_threshold=0.0009,
                             transaction_cost=0.0002, leverage=8.33)
        fig = plt.figure(figsize=(24,12))
        ax1 = fig.add_subplot(111)
        ax1.plot(df['trade_day'], round(df['market_price_close']  ), color='orange', label='沪深300股指期货(左轴, 元)')
        ax1.legend(loc='lower left', bbox_to_anchor=(0.3, -0.2) )
        ax1.set_ylim([min(df['market_price_close']) - 200, max(df['market_price_close']) + 200])
        ax2 = ax1.twinx()
        ax2.plot(df['trade_day'], round(df['return']*100,2), color = 'blue', label='模型累计收益(右轴, %)')
        ax2.legend(loc='lower right', bbox_to_anchor=(0.7, -0.2) )
        ax2.set_ylim([ min(df['return'])*110 , max(df['return'])*110 ])
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=30))
        fig.autofmt_xdate()
        plt.savefig('./图9 平稳度指数交易模型资产累计收益.png')
        print('图9做好了！')

    def figure_11(self):
      '''
        input 函数输入参数 无
        output 函数输出参数 无
      '''
      fig = plt.figure(figsize=(24, 12))
      ax1 = fig.add_subplot(111)
      for i in range(7,12):
        df = self.single_strategy(obs_end_time=50, cut_loss_threshold=0.005, emotion_stationary_threshold=i/10000,
                             transaction_cost=0.0002, leverage=8.33)
        ax1.plot(df['trade_day'], round(df['return']*100,2), label=str(i)+'/10000')
        ax1.legend(loc='upper left')
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=30))
        fig.autofmt_xdate()
        plt.savefig('./图11 不同开仓阈值条件下资产累计收益.png')
      print('图11做好了！')

    def figure_12(self):
      '''
        input 函数输入参数 无
        output 函数输出参数 无
      '''
      fig = plt.figure(figsize=(24, 12))
      ax1 = fig.add_subplot(111)
      for i in range(48,53):
        df = self.single_strategy(obs_end_time=i, cut_loss_threshold=0.005, emotion_stationary_threshold=9/10000,
                             transaction_cost=0.0002, leverage=8.33)
        ax1.plot(df['trade_day'], round(df['return']*100,2), label='早上开盘'+str(i)+'分钟后结束观察期')
        ax1.legend(loc='upper left')
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=30))
        fig.autofmt_xdate()
        plt.savefig('./图12 不同开仓时间条件下资产累计收益.png')
      print('图12做好了！')

    def multi_strategy(self, obs_end_time=50, obs_end_time_afternoon=32,  cut_loss_threshold=0.005, emotion_stationary_threshold=0.0009,transaction_cost=0.0002, leverage = 8.33):
         '''

         input 函数输入参数
         obs_end_time  int   某天上午开盘后到观察期结束的分钟数 默认50
         obs_end_time_afternoon  int   某天下午观察期结束时是13点的哪一分钟 默认32
         cut_loss_threshold float 止损阙值 默认0.005
         emotion_stationary_threshold float 情绪平稳度阙值 默认0.0009
         transaction_cost float 双边交易成本 默认0.0002
         leverage float 杠杆 默认8.33
         output 函数输出参数
         res_df dataframe columns: tradeday str 交易日, emotion_stationary float 首次观察期市场平稳度 ,
          market_price_close float 沪深300日收盘价,  net_equity float 净资产, return 收益率
         '''
         # process input data
         data = pd.DataFrame({'trade_day':self.data['trade_time'].apply(lambda x: str(x)[:10]), 'trade_time': self.data['trade_time'].apply(lambda x: str(x)[11:]),
                              'open': self.data['open'] , 'close': self.data['close'] , 'high': self.data['high'], 'low': self.data['low'],
                              'daily_obs_max_drawdown_rate': [np.nan]*len(self.data['trade_time']),'net_equity': [1000000000]*len(self.data['trade_time'])
                              } )

         # 初始化结果中需要的参数
         res = [] #二位列表
         # 计算结果
         for index, row in data.iterrows():
             # 按论文要求，跳过9:30之前的盘前和15:00之后的盘后的数据
             if (row['trade_time'][:2] == '09' and float(row['trade_time'][3:5]) < 30)  \
                     or  (row['trade_time'][:2] == '15' and float(row['trade_time'][3:5]) >= 1)    :
                 continue
             #--------------------------------------------遍历开盘到收盘------------------------------------------------

             #--------------------------------------------第一阶段------------------------------------------------
             if row['trade_time'] == '09:30:00': #9:30新的交易日开盘了！初始化各种参数
                 v,v_2 = float('-inf'),float('inf') #初始化沪深三百价格(v用作做多,v_2用作做空）
                 m,m_2 = 0,0 #初始化最大回撤为0(m用作做多,m_2用作做空）
                 lst,lst_2 = [], [] #初始化最大回撤列表（lst用作做多,lst_2用作做空）
                 open = row['open'] #当日开盘价
                 is_open = [0,None,None]  #初始化判断是否开仓(0不开，1开） , 方向（0做空，1做多）, 开仓价位
                 is_close = [0] #初始化判断日内是否平仓过至少一次（0没有，1大于一次）
                 is_open_afternoon = [0,None,None]  #初始化判断下午阶段是否开仓(0不开，1开） , 方向（0做空，1做多）, 开仓价位
                 is_close_afternoon = [0] #初始化判断日内下午阶段是否平仓过至少一次（0没有，1大于一次）
                 is_obs = 1  #初始化判断是否处于上午观察期（0不处于，1处于）
                 is_obs_afternoon = 0  #初始化判断是否处于下午观察期（0不处于，1处于）
                 emotion_stationary = float('inf') #初始化上午平稳度
                 emotion_stationary_afternoon = float('inf')  # 初始化下午平稳度
                 net_equity = 1000000000 if res == [] else res[-1][-2] #初始化净资产

             if is_obs == 1: #判断是否在观察期，处于观察期判断回撤
               v = max(v,  row['high'])          #日内最高价（剔除未来函数）
               m = max(1 - row['low'] / v, m, 0) #日内最大回撤（做多，剔除未来函数）
               v_2 = min(v,  row['low'])  #日内最低价（剔除未来函数）
               m_2 = max(row['high']/ v_2 - 1, m_2,0) #日内最大回撤（做空，剔除未来函数）
               lst.append(m)
               lst_2.append(m_2)
             if is_obs == 1 and len(lst) == obs_end_time: #判断观察期是否到了结束时刻，结束时计算平稳度，收盘价到收盘才更新，更新别的项
                 emotion_stationary = min(sum(lst)/len(lst),sum(lst_2)/len(lst_2))
                 res.append([row['trade_day'], emotion_stationary, np.nan,net_equity,net_equity/1000000000-1])
                 is_obs = 0
             if is_obs == 0 and emotion_stationary <= emotion_stationary_threshold and is_open[0] == 0 and is_close[0] == 0: #观察期结束后在没有开仓下判断是否开仓
                 if open < row['close']:
                     is_open[0], is_open[1],is_open[2] = 1, 1, row['close']
                     net_equity +=  -transaction_cost*net_equity*leverage
                 if open > row['close']:
                     is_open[0], is_open[1],is_open[2] = 1, 0, row['close']
                     net_equity +=  -transaction_cost*net_equity*leverage
             if is_open[0] == 1: #开仓后判断是否止损或收盘平仓
                 if is_open[1] == 1 and (row['low'] - is_open[2])/is_open[2] <= -cut_loss_threshold: #做多止损平仓
                     net_equity += ( row['low']-is_open[2])/is_open[2]*leverage*net_equity -  leverage * net_equity * transaction_cost
                     res[-1][-2] = net_equity
                     is_open[0] = 0
                     is_close[0] = 1
                 if is_open[1] == 0 and (row['high'] - is_open[2])/is_open[2] >= cut_loss_threshold: #做空止损平仓
                     net_equity +=  -(row['high'] - is_open[2])/is_open[2] * leverage *net_equity -  leverage * net_equity * transaction_cost
                     res[-1][-2] = net_equity
                     is_open[0] = 0
                     is_close[0] = 1
                 if is_open[1] == 1 and row['trade_time'] == '11:30:00' : #做多收盘平仓
                     net_equity +=  ((row['close'] - is_open[2])/is_open[2]) * leverage * net_equity -  leverage * net_equity * transaction_cost
                     res[-1][-2] = net_equity
                     res[-1][-2] = net_equity
                     is_open[0] = 0
                     is_close[0] = 1
                 if is_open[1] == 0 and row['trade_time'] == '11:30:00' : #做空收盘平仓
                     net_equity +=  ((is_open[2]- row['close'])/is_open[2]) * leverage * net_equity  -  leverage * net_equity * transaction_cost
                     res[-1][-2] = net_equity
                     is_open[0] = 0
                     is_close[0] = 1
                 res[-1][-1] = res[-1][-2]/1000000000-1

             # --------------------------------------------第二阶段------------------------------------------------
             if row['trade_time'] == '11:12:00': #下午观察期开始了！初始化各种参数
                 v_afternoon, v_2_afternoon = float('-inf'),float('inf') #初始化沪深三百价格(v_afternoon用作做多,v_2_afternoon用作做空）
                 m_afternoon,m_2_afternoon = 0,0 #初始化最大回撤为0(m_afternoon用作做多,m_2_afternoon用作做空）
                 lst_afternoon,lst_2_afternoon = [], [] #初始化最大回撤列表（lst_afternoon用作做多,lst_2_afternoon用作做空）
                 open_afternoon = row['open'] #当日下午观察期第一个价
                 is_obs_afternoon = 1  #初始化判断是否处于观察期（0不处于，1处于）
             if is_obs_afternoon == 1: #判断是否在观察期，处于观察期判断回撤
               v_afternoon = max(v_afternoon,  row['high'])          #日内最高价（剔除未来函数）
               m_afternoon = max(1 - row['low'] / v_afternoon, m_afternoon, 0) #日内最大回撤（做多，剔除未来函数）
               v_2_afternoon = min(v_afternoon,  row['low'])  #日内最低价（剔除未来函数）
               m_2_afternoon = max(row['high']/ v_2_afternoon - 1, m_2_afternoon,0) #日内最大回撤（做空，剔除未来函数）
               lst_afternoon.append(m_afternoon)
               lst_2_afternoon.append(m_2_afternoon)
             if is_obs_afternoon == 1 and row['trade_time'] == '13:'+ str(obs_end_time_afternoon) +':00': #判断观察期是否到了结束时刻，结束时计算平稳度，收盘价到收盘才更新，更新别的项
                 emotion_stationary_afternoon = min(sum(lst_afternoon)/len(lst_afternoon),sum(lst_2_afternoon)/len(lst_2_afternoon))
                 is_obs_afternoon = 0
             if is_obs_afternoon == 0 and emotion_stationary_afternoon <= emotion_stationary_threshold and is_open_afternoon[0] == 0 and is_close_afternoon[0] == 0: #观察期结束后在没有开仓下判断是否开仓
                 if open_afternoon < row['close']:
                     is_open_afternoon[0], is_open_afternoon[1],is_open_afternoon[2] = 1, 1, row['close']
                     net_equity +=  -transaction_cost*net_equity*leverage
                 if open_afternoon > row['close']:
                     is_open_afternoon[0], is_open_afternoon[1],is_open_afternoon[2] = 1, 0, row['close']
                     net_equity +=  -transaction_cost*net_equity*leverage
             if is_open_afternoon[0] == 1: #开仓后判断是否止损或收盘平仓
                 if is_open_afternoon[1] == 1 and (row['low'] - is_open_afternoon[2])/is_open_afternoon[2] <= -cut_loss_threshold: #做多止损平仓
                     net_equity += ( row['low']-is_open_afternoon[2])/is_open_afternoon[2]*leverage*net_equity -  leverage * net_equity * transaction_cost
                     res[-1][-2] = net_equity
                     is_open_afternoon[0] = 0
                     is_close_afternoon[0] = 1
                 if is_open_afternoon[1] == 0 and (row['high'] - is_open_afternoon[2])/is_open_afternoon[2] >= cut_loss_threshold: #做空止损平仓
                     net_equity +=  -(row['high'] - is_open_afternoon[2])/is_open_afternoon[2] * leverage *net_equity -  leverage * net_equity * transaction_cost
                     res[-1][-2] = net_equity
                     is_open_afternoon[0] = 0
                     is_close_afternoon[0] = 1
                 if is_open_afternoon[1] == 1 and row['trade_time'] == '15:00:00' : #做多收盘平仓
                     net_equity +=  ((row['close'] - is_open_afternoon[2])/is_open_afternoon[2]) * leverage * net_equity -  leverage * net_equity * transaction_cost
                     res[-1][-2] = net_equity
                     res[-1][-2] = net_equity
                     is_open_afternoon[0] = 0
                     is_close_afternoon[0] = 1
                 if is_open_afternoon[1] == 0 and row['trade_time'] == '15:00:00' : #做空收盘平仓
                     net_equity +=  ((is_open_afternoon[2]- row['close'])/is_open_afternoon[2]) * leverage * net_equity  -  leverage * net_equity * transaction_cost
                     res[-1][-2] = net_equity
                     is_open_afternoon[0] = 0
                     is_close_afternoon[0] = 1
                 res[-1][-1] = res[-1][-2]/1000000000-1

             if  row['trade_time'] == '15:00:00' : #收盘时无论今日有无交易，更新结果中的收盘价
                 res[-1][2] = row['close']

         res_df = pd.DataFrame(res,columns=['trade_day','emotion_stationary','market_price_close','net_equity','return'])

         return  res_df

    def figure_17(self):
        '''
        input 函数输入参数 无
        output 函数输出参数 无
        '''
        df = self.multi_strategy(obs_end_time=50, obs_end_time_afternoon=32, cut_loss_threshold=0.005, emotion_stationary_threshold=0.0009,
                             transaction_cost=0.0002, leverage=8.33)
        fig = plt.figure(figsize=(24,12))
        ax1 = fig.add_subplot(111)
        ax1.plot(df['trade_day'], round(df['market_price_close']  ), color='orange', label='沪深300股指期货(左轴, 元)')
        ax1.legend(loc='lower left', bbox_to_anchor=(0.3, -0.2) )
        ax1.set_ylim([min(df['market_price_close']) - 200, max(df['market_price_close']) + 200])
        ax2 = ax1.twinx()
        ax2.plot(df['trade_day'], round(df['return']*100,2), color = 'blue', label='模型累计收益(右轴, %)')
        ax2.legend(loc='lower right', bbox_to_anchor=(0.7, -0.2) )
        ax2.set_ylim([ min(df['return'])*110 , max(df['return'])*110 ])
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=30))
        fig.autofmt_xdate()
        plt.savefig('./图17 多次开仓市场平稳度模型累计收益率.png')

        print('图17做好了！')

    def figure_19(self):
        '''
        input 函数输入参数 无
        output 函数输出参数 无
        '''
        '''
        input 函数输入参数 无
        output 函数输出参数 无
        '''
        df = self.single_strategy(obs_end_time=50, cut_loss_threshold=0.005, emotion_stationary_threshold=0.0009,
                             transaction_cost=0.0002, leverage=8.33)
        df2 = self.multi_strategy(obs_end_time=50, obs_end_time_afternoon=32,  cut_loss_threshold=0.005, emotion_stationary_threshold=0.0009,transaction_cost=0.0002, leverage = 8.33)
        fig = plt.figure(figsize=(24,12))
        ax1 = fig.add_subplot(111)
        ax1.plot(df['trade_day'], round(df['market_price_close']  ), color='silver', label='沪深300股指期货(左轴, 元)')
        ax1.legend(loc='lower left', bbox_to_anchor=(0.3, -0.2) )
        ax1.set_ylim([min(df['market_price_close']) - 200, max(df['market_price_close']) + 200])
        ax2 = ax1.twinx()
        ax2.plot(df['trade_day'], round(df['return']*100,2), color = 'orange', label='单次开仓策略（右轴,%）')
        ax2.plot(df2['trade_day'], round(df2['return'] * 100, 2), color='blue', label='多次开仓策略（右轴,%）')
        ax2.legend(loc='lower right', bbox_to_anchor=(0.7, -0.2) )
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=30))
        fig.autofmt_xdate()
        plt.savefig('./图19 多次开仓策略与原始策略累计收益率对比.png')

        print('图19做好了！')

    def figure_20(self):
      '''
        input 函数输入参数 无
        output 函数输出参数 无
      '''
      fig = plt.figure(figsize=(24, 12))
      ax1 = fig.add_subplot(111)
      for i in range(48,53):
        df = self.multi_strategy(obs_end_time=i, obs_end_time_afternoon=32, cut_loss_threshold=0.005, emotion_stationary_threshold=9/10000,
                             transaction_cost=0.0002, leverage=8.33)
        ax1.plot(df['trade_day'], round(df['return']*100,2), label='早上开盘'+str(i)+'分钟后结束观察期')
        ax1.legend(loc='upper left')
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=30))
        fig.autofmt_xdate()
        plt.savefig('./图20 上午不同时间开仓.png')
      print('图20做好了！')

    def figure_21(self):
      '''
        input 函数输入参数 无
        output 函数输出参数 无
      '''
      fig = plt.figure(figsize=(24, 12))
      ax1 = fig.add_subplot(111)
      for i in range(31,36):
        df = self.multi_strategy(obs_end_time=50, obs_end_time_afternoon=i, cut_loss_threshold=0.005, emotion_stationary_threshold=9/10000,
                             transaction_cost=0.0002, leverage=8.33)
        ax1.plot(df['trade_day'], round(df['return']*100,2),label='下午开盘'+str(i)+'分钟后结束观察期')
        ax1.legend(loc='upper left')
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=30))
        fig.autofmt_xdate()
        plt.savefig('./图21 下午不同时间开仓.png')
      print('图21做好了！')

    def figure_22(self):
      '''
        input 函数输入参数 无
        output 函数输出参数 无
      '''
      fig = plt.figure(figsize=(24, 12))
      ax1 = fig.add_subplot(111)
      for i in range(7,12):
        df = self.multi_strategy(obs_end_time=50, obs_end_time_afternoon=32, cut_loss_threshold=0.005, emotion_stationary_threshold=i/10000,
                             transaction_cost=0.0002, leverage=8.33)
        ax1.plot(df['trade_day'], round(df['return']*100,2), label=str(i)+'/10000')
        ax1.legend(loc='upper left')
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=30))
        fig.autofmt_xdate()
        plt.savefig('./图22 不同开仓阈值条件下资产累计收益.png')
      print('图22做好了！')

if __name__ == "__main__": #测试用
     test = Create_Return()
     #print(test.data)
     #print( test.multi_strategy(obs_end_time=50, obs_end_time_afternoon=32,  cut_loss_threshold=0.005,emotion_stationary_threshold=0.0009,transaction_cost=0.0002, leverage = 8.33) )
     test.figure_9()
     test.figure_11()
     test.figure_12()
     test.figure_17()
     test.figure_19()
     test.figure_20()
     test.figure_21()
     test.figure_22()


