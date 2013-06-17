import json
import flask
import datetime
import sys

from collections import Counter
import statsmodels.api as sm
import numpy as np
import matplotlib.pylab as plt


# input is the json data and output is x (a list of datetimes) and y (a list of counts for demand each hour)
def process_input(json_data):
    raw_data = json.load(json_data)
    data = [datetime.datetime.strptime(s[:-12], '%Y-%m-%dT%H') for s in raw_data]
    first = min(data) # earliest datetime
    last = max(data) # latest datetime
    total_hours = (last-first).days*24 + (last-first).seconds//3600
    counts = Counter(data)
    # x is array of datetime objects for the time period covered and y are the counts of logins
    x = [first + datetime.timedelta(hours=k) for k in range(0,total_hours+1)]
    y = [counts[k] for k in x]
    return x, y, last, total_hours

# input is a list x, a lists of lists y_array, and a list of strings as the labels
# Each list in y_array will be plotted against x
def plot_series(x, y_array, labels):
    for y_arr, label in zip(y_array, labels):
        plt.plot(x, y_arr, label=label)
        plt.xlabel('Datetime')
        plt.ylabel('Demand')
        plt.title('Models of demand using trends and ARMA')
    plt.gcf().set_size_inches(26,20)
    plt.legend()
    plt.show()

# input is two lists and returns the root mean square error
def rms(y, predict):
    diff = [w1 - w2 for (w1,w2) in zip(y, predict)]
    return np.sqrt(np.mean([x**2 for x in diff]))

def main():
    # Process input data
    #json_data=open('C:/Users/rthomas/Documents/DemandPrediction/demand_prediction.json')
    json_data = open(sys.argv[1])
    x, y, last, total_hours = process_input(json_data)
    FUTURE_DAYS = 15  # will make prediciton 15 days into future

    # I looked at a few different regression families in sm.GLM but found very similar rms errors so I chose to use a simple linear regression
    trend_model = sm.OLS(y, sm.add_constant(range(len(x)), prepend=True)).fit()
    trend = trend_model.fittedvalues

    # y1 is y with the trend line (growth over time) removed
    y1 = y - trend
    
    # y2 is y1 with weekly trends removed
    days = [w.weekday() for w in x]
    days_mean = [np.mean([y1[i] for i in range(0,len(y1)) if days[i] == k]) for k in range(7)]
    y2 = [y1[i] - days_mean[days[i]] for i in range(len(y1))]
    
    # y3 is y2 with daily trends removed
    hours = [w.hour for w in x]
    hours_mean = [np.mean([y2[i] for i in range(0,len(y2)) if hours[i] == k]) for k in range(24)]
    y3 = [y2[i] - hours_mean[hours[i]] for i in range(len(y2))]

    trend_y = [days_mean[days[i]] + hours_mean[hours[i]] + trend[i] for i in range(len(trend))]

    # The predict function is not working the way I expected for future times
    # construct an ARMA model on the residuals once all trends have been removed
    #arma_model = sm.tsa.ARMA(y3)
    #arma_fit = arma_model.fit(order=(6,6))
    #arma_res = arma_fit.predict()
    #arma_y = [arma_res[i] + trend_y[i] for i in range(len(trend))]
    
    future_hours = FUTURE_DAYS*24 + total_hours
    future_trend = trend_model.predict(sm.add_constant(range(total_hours, future_hours), prepend=True))
    future_x = [last + datetime.timedelta(hours=k) for k in range(1,FUTURE_DAYS*24+1)]
    future_hours = [w.hour for w in future_x]
    future_days = [w.weekday() for w in future_x]
    future_hours_trend = [hours_mean[future_hours[i]] for i in range(len(future_x))]
    future_days_trend = [days_mean[future_days[i]] for i in range(len(future_x))]
    future_all_trends = [sum(tuple) for tuple in zip(future_trend, future_hours_trend, future_days_trend)] 
    
    
    

    #plot_series(x, [y, trend_y, arma_y], ['Time Series', 'All trends', 'All trends + ARMA'])

    #rms_arr = []
    #for predict in [trend_y, arma_y]:
    #    rms_arr.append(rms(y, predict))
            
    app = flask.Flask(__name__)
    app.run()
            
            

if __name__ == '__main__':
    main()


#app.config['DEBUG'] = True





