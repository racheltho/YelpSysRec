import sys
import json
import re
import csv
import pandas as pd
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse

 
def rms(y, predict):
    diff = [w1 - w2 for (w1,w2) in zip(y, predict)]
    return np.sqrt(np.mean([x**2 for x in diff])) 
  
def main():
    training_directory = "C:/Users/rthomas/Desktop/Kaggle/Sysrec/yelp_training_set"
    business_train = pd.read_csv(training_directory + "/yelp_training_set_business.csv")
    user_train = pd.read_csv(training_directory + "/yelp_training_set_user.csv")
    checkin_train = pd.read_csv(training_directory + "/yelp_training_set_checkin.csv")
    review_train = pd.read_csv(training_directory + "/yelp_training_set_review.csv")
    test_directory = "C:/Users/rthomas/Desktop/Kaggle/Sysrec/yelp_test_set"
    business_test = pd.read_csv(test_directory + "/yelp_test_set_business.csv")
    user_test = pd.read_csv(test_directory + "/yelp_test_set_user.csv")
    checkin_test = pd.read_csv(test_directory + "/yelp_test_set_checkin.csv")
    review_test = pd.read_csv(test_directory + "/yelp_test_set_review.csv")
    business_all = pd.concat([business_train, business_test])
    user_all = pd.concat([user_train, user_test])
    
    r = len(review_test)
    r_perm = np.random.permutation(range(r))
    r_size = round(r*.4)
    r_train = review_train.ix[r_perm[:r_size]]
    r_cv = review_train.ix[r_perm[r_size:]]

    train_data = pd.merge(pd.merge(r_train, user_train, how='left', on='user_id'), business_train, how='left', on='business_id')
    #cv_data = pd.merge(pd.merge(r_cv, user_train, how='left', on='user_id'), business_train, how='left', on='business_id')

    #mask_b = business_train['business_id'].map(lambda x: x in r_train['business_id'].values)
    #train_b  = business_train[mask_b]
    #mask_u = user_train['user_id'].map(lambda x: x in r_train['user_id'].values)
    #train_u = user_train[mask_u]

    mask_b = business_all['business_id'].map(lambda x: x in review_test['business_id'].values)
    test_b  = business_all[mask_b]
    mask_u = user_all['user_id'].map(lambda x: x in review_test['user_id'].values)
    test_u = user_all[mask_u]

    datu = test_u['average_stars'].values
    mdatu = np.ma.masked_array(datu,np.isnan(datu))
    avg_stars_u = np.mean(mdatu)
    datb = test_b['stars'].values
    mdatb = np.ma.masked_array(datb,np.isnan(datb))
    avg_stars_b = np.mean(mdatb)

    test_pairs = [tuple(x) for x in review_test[['user_id','business_id']].values]
    #cv_pairs = [tuple(x) for x in cv_data[['user_id','business_id']].values]
    #cv_stars = cv_data['stars_x'].values

    with open('C:/Users/rthomas/Desktop/Kaggle/Sysrec/mybenchmark.csv', 'wb') as csvfile:
        mywriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        mywriter.writerow(['user_id', 'business_id', 'stars'])
        predictions = []
        for t in test_pairs:
            u = test_u[ test_u['user_id']==t[0] ]['average_stars'].values
            b = test_b[ test_b['business_id']==t[1] ]['stars'].values
            if len(u) and len(b) and not np.isnan(u[0]) and not np.isnan(b[0]):
                p = (u[0] - avg_stars_u) + b[0]
            elif len(b) and not np.isnan(b[0]):
                p = b[0]
            elif len(u) and not np.isnan(u[0]):
                p = (u[0] - avg_stars_u) + avg_stars_b
            else:
                p = avg_stars_b
            p = min(p,5)
            p = max(p,1)
            mywriter.writerow([t[0], t[1], p])

    
if __name__ == '__main__':
    main()
