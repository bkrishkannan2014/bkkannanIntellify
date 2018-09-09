# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 06:23:05 2018

@author: Kannan
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse
from datetime import date, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as dates

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

def get_pred_error(train, test):
    ldata = get_samplesbylabel(train)
    avg=list(map(lambda x: np.mean(x), ldata))
    mse = []
    for ii in range(len(test)):
        tlab = test['kmlabel'].values[ii]
        mse.append(np.abs(test['travel_time'].values[ii]-avg[tlab]))
        
    return mse

def get_pred_error2(train, test, ipastdays):
    ldata = get_samplesbylabel(train)
    mse = []
    for ii in range(len(test)):
        tmp=train[(train['dtobj']<=test['dtobj'].values[ii]) & (train['kmlabel']==test['kmlabel'].values[ii]) & 
                  (train['dtobj']>test['dtobj'].values[ii]-ipastdays*3600*24)]
        pvals = tmp['travel_time']
        mse.append(np.abs(test['travel_time'].values[ii]-np.mean(pvals)))
        
    return mse
def get_samplesbylabel(train):
    ulabs = np.unique(train['kmlabel'].values)
    
    ldata = []
    for ul in ulabs:
        iz = np.where(train['kmlabel'].values==ul)
        iz = iz[0]
        
        ldata.append(list(train['travel_time'].values[iz]))
    return ldata
    
def seperate_samplesby_wd_3mins(data, uday):
    utime = np.unique(data['totsecs'].values)
    
    ldata = []
    for id in utime:
        iz = np.where((data['Weekday'].values==uday) & (data['totsecs'].values==id))
        iz = iz[0]
        
        ldata.append(list(data['travel_time'].values[iz]))
    return ldata
    
    
    
def get_school_holidays():
    
    dhols = []
    holidays = pd.read_csv('schoolholidays.csv', index_col=None)
    dts      = [datetime.strptime(twd, '%m/%d/%Y') for twd in holidays['start'].values]
    dte      = [datetime.strptime(twd, '%m/%d/%Y') for twd in holidays['end'].values]
    
    for id in range(0, len(dts)):
        delta = dte[id] - dts[id]
        for jd in range(delta.days+1):
            dhols.append(dts[id] + timedelta(jd))
    return dhols

def get_public_holidays():

    holidays = pd.read_csv('publicholidyas.csv', index_col=None)
    dts      = [datetime.strptime(twd, '%m/%d/%Y') for twd in holidays['date'].values]
    return dts

def assign_public_schoolholidayclass(data):
      ph = get_public_holidays()
      sh = get_school_holidays()
      dt      = [datetime.strptime(twd, '%m/%d/%Y') for twd in data['Date'].values]
      i=0
      for idt in dt:
          
          if idt in ph:
              data['Weekday'].values[i] = 2
              
          if idt in sh:
              if data['Weekday'].values[i] !=1:
                  data['Weekday'].values[i] = 3
              
          i= i+ 1
      return data
  
def plot_hist(inData):
    
    for iC in range(0,3):
        iz = np.where(histdata_wdays['Peakhour'].values==iC)
        plt.hist(histdata_wdays['travel_time'].values[iz[0]], bins=50)
    
def plot_days(inData):
    
    udays = np.unique(inData['Date'].values)
    for iud in udays:
        print(iud)
        iz = np.where(inData['Date'].values == iud)
        dataplot = inData['travel_time'].values[iz[0]]
        dt    = [datetime.strptime(twd, '%H:%M:%S') for twd in inData['Time'].values[iz[0]]]
        ttmd      = dates.date2num(dt)
        plt.plot(ttmd, dataplot)
        
    myFmt = dates.DateFormatter('%H:%M:%S')
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.gcf().autofmt_xdate()
 
# unique days

#http://radiostud.io/beat-rush-hour-traffic-with-tensorflow-machine-learning/
filename = '40010.csv'
data = pd.read_csv(filename, index_col=None)

###assign weekday and weekend index
wdstr = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']
wd_end = [1,0,0,0,0,0,1]
wd     = [datetime.strptime(twd, '%m/%d/%Y').strftime('%a') for twd in data['Date'].values]
wdC    = [wd_end[wdstr.index(iwd)] for iwd in wd]
data['Weekday'] = wdC

# assign shole holidays

#Assign peak/normal/nights periods classes
pt     = [datetime.strptime(x,'%H:%M:%S') for x in data['Time'].values]
total_seconds = [ipt.second+ipt.minute*60+ipt.hour*3600 for ipt in pt]
data['totsecs'] = total_seconds

#peakhours
Pkhour1_S = 0 + 30*60+7*3600
Pkhour1_E = 0 + 30*60+9*3600

Pkhour2_S = 0 + 30*60+14*3600
Pkhour2_E = 0 + 0*60+19*3600

#night hour
Nghthour1_S  = 0 + 0*60+19*3600
Nghthour1_E = 0 + 60*60+23*3600

Nghthour2_S  = 0 + 0*60+0*3600
Nghtkhour2_E = 0 + 30*60+7*3600

# normal hour
Norhour_S = 0 + 30*60+9*3600
Norhour_E = 0 + 30*60+14*3600


PHC = []

for it in total_seconds:
    if (it>=0) & (it<Pkhour1_S):
        PHC.append(0)
    elif (it>=Pkhour1_S) & (it<=Pkhour1_E):
        PHC.append(1)
    elif (it>Pkhour1_E) & (it<Pkhour2_S):
        PHC.append(2)
    elif (it>=Pkhour2_S) & (it<=Pkhour2_E):
        PHC.append(3)
    else:
        PHC.append(0)

data['Peakhour'] = PHC
####
# convert travel_time to floats
# assign -1 to unknow travel time 
izX = np.where(data['travel_time'].values=='x')
izX = izX[0]
data['travel_time'].values[izX] = -1
data['travel_time'] = np.asarray([float(x) for x in data['travel_time'].values])



#combine date and time
dtstr = data['Date'].values+ ' ' + data['Time'].values
data['datetime']     = dtstr #[datetime.strptime(twd, '%Y-%m-%d %H:%M:%S') for twd in dtstr]
data['dtobj'] = [datetime.strptime(twd, '%m/%d/%Y %H:%M:%S').timestamp() for twd in dtstr]
#ttmd      = dates.date2num(data['datetime'].values)

#separete the hist data with travel time and the data to be predicted (with -1)
iznX = np.where(data['travel_time'].values!=-1)
iznX = iznX[0]
histdata = data.iloc[iznX]

iznX = np.where(data['travel_time'].values==-1)
iznX = iznX[0]
preddata = data.iloc[iznX]

#missing values: remove the missing values
missing_nan     = np.isnan(histdata['travel_time'].values)
missing_nan_ind = np.where(missing_nan==False)
histdata = histdata.iloc[missing_nan_ind[0]]

# let's do some cploratory analysis
# separate week and weekends days
iz = np.where(histdata['Weekday'].values==0)
histdata_wdays = histdata.iloc[iz[0]]
iz = np.where(histdata['Weekday'].values==1)
histdata_wends = histdata.iloc[iz[0]]
# unique days
ud_wdays = np.unique(histdata_wdays['Date'].values)
ud_wend = np.unique(histdata_wends['Date'].values)

##convert to datetime
#dtobj = datetime.strptime(data['Date'].values[0], '%Y-%m-%d')
#

#
#histdata1['PeakHour'] = PHC      
##combine date and time
#dtstr = histdata1['Date'].values+' ' +histdata1['Time'].values
#dt    = [datetime.strptime(twd, '%Y-%m-%d %H:%M:%S') for twd in dtstr]
#ttmd      = dates.date2num(dt)

#plt.gcf().autofmt_xdate()
#myFmt = dates.DateFormatter('%H:%M')
#plt.gca().xaxis.set_major_formatter(myFmt)
#
#plt.show()
# get the summary stats
        
            
##df.boxplot(by='Peakhour')
#df=histdata[['Weekday','travel_time','Peakhour']]
#
#df.boxplot(by=['Weekday','Peakhour'])
#avg=list(map(lambda x: np.mean(x), ldata))

hd = assign_public_schoolholidayclass(histdata)

#y = hd.travel_time
#X_train, X_test, y_train, y_test = train_test_split(hd, y, test_size=0.33)
msk = np.random.rand(len(hd)) < 0.8
train = hd[msk]
test  = hd[~msk]

kmmdl = KMeans(n_clusters = 4*4, random_state = 2).fit(train[['Weekday','Peakhour']])
train['kmlabel'] = kmmdl.labels_

ppredi = kmmdl.predict(test[['Weekday','Peakhour']])
test['kmlabel'] = ppredi

#tmp=train[(train['dtobj']<=test['dtobj'].values[100]) & (train['kmlabel']==test['kmlabel'].values[100])]
#tmp=train[(train['dtobj']<=test['dtobj'].values[100]) & (train['kmlabel']==test['kmlabel'].values[100]) & (train['dtobj']>test['dtobj'].values[100]-3*3600*24)]