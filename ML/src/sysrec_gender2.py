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
    all_data = pd.merge(pd.merge(review_all, user_all, how='left', on="user_id"),business_all, how='left', on="business_id")
    
    review_test_ext = pd.merge(pd.merge(review_test, user_all, how='left', on='user_id'),business_all, how='left', on='business_id')
    review_test_ext.sort(['order'], inplace=True)
    
    # Averages by (business_id, gender) pairs
    rte_mask = all_data['stars_x'].map(lambda x: not np.isnan(x))
    groups = all_data[rte_mask].groupby(['business_id','mf'])
    group_counts = groups['user_id'].count()
    group_stars = groups['stars_x'].mean()
    bus_gen_dict = {group_stars.index[i]: group_stars[i] for i in range(len(group_stars))}
    count_dict = {group_counts.index[i]: group_counts[i] for i in range(len(group_counts))}
    
    #user_review_groups = all_data[rte_mask].groupby(['user_id'])
    #user_review_counts = user_review_groups['stars_x'].count()
    #user_review_dict = {user_review_counts.index[i]: user_review_counts[i] for i in range(len(user_review_counts))}
    
    # Averages for all male users and all female users
    mask_mf = user_all['user_id'].map(lambda x: x in review_test['user_id'].values)
    test_mf = user_all[mask_mf]
    groupsmf = test_mf.groupby(['mf'])
    group_stars_mf = groupsmf['average_stars'].mean()
    
    #groupsmf_other = review_test_ext.groupby(['mf'])
    #group_stars_mf_other = groupsmf_other['stars'].mean()
    
    #datu2 = review_test_ext['average_stars'].values
    #avg_stars_all_u = np.mean(np.ma.masked_array(datu2,np.isnan(datu2)))
    #datb2 = review_test_ext['stars'].values
    #avg_stars_all_b = np.mean(np.ma.masked_array(datb2,np.isnan(datb2)))


    # this is what I did in my benchmark- not sure why it gives a better answer?
    mask_b = business_all['business_id'].map(lambda x: x in review_test['business_id'].values)
    test_b  = business_all[mask_b]
    mask_u = user_all['user_id'].map(lambda x: x in review_test['user_id'].values)
    test_u = user_all[mask_u]
    datu = test_u['average_stars'].values
    mdatu = np.ma.masked_array(datu,np.isnan(datu))
    old_avg_stars_u = np.mean(mdatu)
    datb = test_b['stars'].values
    mdatb = np.ma.masked_array(datb,np.isnan(datb))
    old_avg_stars_b = np.mean(mdatb)

    # Averages by (business_id, gender) pairs
    rte_mask = all_data['stars_x'].map(lambda x: not np.isnan(x))
    groups = all_data[rte_mask].groupby(['business_id','mf'])
    group_counts = groups['user_id'].count()
    group_stars = groups['stars_x'].mean()
    bus_gen_dict = {group_stars.index[i]: group_stars[i] for i in range(len(group_stars))}
    count_dict = {group_counts.index[i]: group_counts[i] for i in range(len(group_counts))}


    test_pairs = [tuple(x) for x in review_test_ext[['user_id','business_id','average_stars','stars', 'mf']].values]

    with open('C:/Users/rthomas/Desktop/Kaggle/Sysrec/genderbenchmark5.csv', 'wb') as csvfile:
        mywriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        # for testing:
        #mywriter.writerow(['avg m', 'avg f', 'old m', 'old f'])
        #mywriter.writerow([group_stars_mf['m'], group_stars_mf['f'], group_stars_mf_other['m'], group_stars_mf_other['f']])
        #mywriter.writerow(['user_id', 'business_id', 'reviews this u', 'stars this u', 'stars this b', 'mf', 'stars', 'what'])
        mywriter.writerow(['user_id', 'business_id', 'stars'])
        for r in test_pairs:
            uid = r[0]
            bid = r[1]
            stars_this_u = r[2]
            stars_this_b = r[3]
            mf = r[4]
            #rev_this_u = ''
            #if uid in user_review_dict.keys():
            #    rev_this_u = user_review_dict[uid]
            # check if we can infer the user's gender
            if mf in ('m','f') and (bid, mf) in bus_gen_dict.keys() and count_dict[(bid, mf)] >= 20:
                # check if there is an average rating by gender for this business (with at least 20 users having rated it)
                if not np.isnan(stars_this_u):
                    stars = bus_gen_dict[(bid, mf)] + (stars_this_u - group_stars_mf[mf])
                    what = 'option 1'
                else:
                    stars = bus_gen_dict[(bid, mf)]
                    what = 'option 2'
                # check if there is an average rating for the business
            elif not np.isnan(stars_this_b):
                # check if there is an average rating for this user
                if not np.isnan(stars_this_u):
                    stars = stars_this_b + (stars_this_u - old_avg_stars_u)
                    what = 'u and b'
                else:
                    stars = stars_this_b
                    what = 'b not u'
                # check if there is an average rating for this user        
            elif not np.isnan(stars_this_u):
                stars = old_avg_stars_b + (stars_this_u - old_avg_stars_u)
                what = 'u not b'
            else:
                stars = old_avg_stars_b
                what = 'neither'
            stars = min(stars,5)
            stars = max(stars,1)
            mywriter.writerow([uid, bid, stars])
            # for testing
            #mywriter.writerow([uid, bid, rev_this_u, stars_this_u, stars_this_b, mf, stars, what])
    
if __name__ == '__main__':
    main()
