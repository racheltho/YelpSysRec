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
    review_test['order'] = review_test.index
    gender = pd.read_table("C:/Users/rthomas/Desktop/Kaggle/Sysrec/mf.txt")

    business_all = pd.concat([business_train, business_test])
    review_all = pd.concat([review_train, review_test])
    user_all = pd.concat([user_train, user_test])
    user_all['name_upper'] = [n.upper() for n in user_all['name']]
    user_all = pd.merge(user_all, gender, how='left', left_on='name_upper', right_on='name')
    all_train = pd.merge(pd.merge(review_train, user_all, how='left', on="user_id"),business_all, how='left', on="business_id")
    review_test_ext = pd.merge(pd.merge(review_test, user_all, how='left', on='user_id'),business_all, how='left', on='business_id')
    review_test_ext.sort(['order'], inplace=True)
    
    # Averages by (business_id, gender) pairs
    groups = all_train.groupby(['business_id','mf'])
    group_counts = groups['user_id'].count()
    group_stars = groups['stars_x'].mean()
    bus_gen_dict = {group_stars.index[i]: group_stars[i] for i in range(len(group_stars))}
    count_dict = {group_counts.index[i]: group_counts[i] for i in range(len(group_counts))}
    
    # Averages for all male users and all female users
    groupsmf = all_train.groupby(['mf'])
    group_stars_mf = groupsmf['stars_x'].mean()
    
    datu2 = user_all['average_stars'].values
    avg_stars_all_u = np.mean(np.ma.masked_array(datu2,np.isnan(datu2)))
    datb2 = business_all['stars'].values
    avg_stars_all_b = np.mean(np.ma.masked_array(datb2,np.isnan(datb2)))

    test_pairs = [tuple(x) for x in review_test_ext[['user_id','business_id','average_stars','stars', 'mf']].values]

    with open('C:/Users/rthomas/Desktop/Kaggle/Sysrec/genderbenchmark2.csv', 'wb') as csvfile:
        mywriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        # for testing:
        #mywriter.writerow(['avg m', 'avg f', 'all u', 'all b'])
        #mywriter.writerow([group_stars_mf['m'], group_stars_mf['f'], avg_stars_all_b, avg_stars_all_u])
        #mywriter.writerow(['user_id', 'business_id', 'stars this u', 'stars this b', 'mf', 'stars', 'what'])
        mywriter.writerow(['user_id', 'business_id', 'stars'])
        for r in test_pairs:
            uid = r[0]
            bid = r[1]
            stars_this_u = r[2]
            stars_this_b = r[3]
            mf = r[4]
            # check if we can infer the user's gender
            if mf in ('m','f'):
                # check if there is an average rating by gender for this business (with at least 22 users having rated it)
                if (bid, mf) in bus_gen_dict.keys() and count_dict[(bid, mf)] >= 22:
                    stars = bus_gen_dict[(bid, mf)]
                    what = 'bus_gen_dict'
                # check if there is an average rating for the business
                elif not np.isnan(stars_this_b):
                    # check if there is an average rating for the user
                    if not np.isnan(stars_this_u):
                        stars = stars_this_b + (stars_this_u - avg_stars_all_u)
                        what = 'option 2'
                    else:
                        stars = stars_this_b
                        what = 'option 3'
                # check if there is an average rating for this user
                elif not np.isnan(stars_this_u):
                    stars = avg_stars_all_b + (stars_this_u - avg_stars_all_u)
                    what = 'option 4'
                # otherwise assign average rating for that gender
                else:
                    stars = group_stars_mf[mf]
                    what = 'option 5'
            else:
                # check if there is an average rating for the business
                if not np.isnan(stars_this_b):
                    # check if there is an average rating for this user
                    if not np.isnan(stars_this_u):
                        stars = stars_this_b + (stars_this_u - avg_stars_all_u)
                        what = 'option 6'
                    else:
                        stars = stars_this_b
                        what = 'option 7'
                # check if there is an average rating for this user        
                elif not np.isnan(stars_this_u):
                    stars = avg_stars_all_b + (stars_this_u - avg_stars_all_u)
                    what = 'option 8'
                else:
                    stars = avg_stars_all_b
                    what = 'option 9'
            stars = min(stars,5)
            stars = max(stars,1)
            mywriter.writerow([uid, bid, stars])
            # for testing
            #mywriter.writerow([uid, bid, stars_this_u, stars_this_b, mf, stars, what])
    
if __name__ == '__main__':
    main()
