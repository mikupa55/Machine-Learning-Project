"""
Date: Mar 10 2017
Program to obtain attributes for bot accounts.
"""

#! /usr/bin/env python

import tweepy
import csv

# Access Credentials for Twitter
CONSUMER_KEY = '93jLsV2DiKbxhposHAoPRzyr4'
CONSUMER_SECRET = '7IVun9JvS6jmfsOmBOBRJ0UDgG1DBchgzyKgbwXoP45Wvb9Wr5'
OWNER = 'sktgater'
OWNER_ID = 152432886
ACCESS_TOKEN = '152432886-1mc2dZw0HTEwsyjbKK4qHJi8fOh0o2OfrDax7eVv'
ACCESS_TOKEN_SECRET = 'sOWblXkALofKyGJr7SYCQcnUMAwvEEDseVPoQ3xAepFox'

# Authentication & API Handler
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Obtain a spreadsheet for each bot account.
# Attributes include: 'id', 'id_str', 'Screen_name', 'Location', 'Description', 'Url', 'Followers_count', 'Friends_count', 'Listed_count', 'Created_at', 'Favourites_count', 'Verified', 'Statuses_count', 'Lang', 'Status', 'Default_profile', 'Default_profile_image','Has_extended_profile','name'

attributes = [    
'id',
'id_str',
'Screen_name',
'Location',
'Description',
'Url',
'Followers_count',
'Friends_count',
'Listed_count',
'Created_at',
'Favourites_count',
'Verified',
'Statuses_count',
'Lang',
'Status',
'Default_profile',
'Default_profile_image',
'Has_extended_profile',
'name',
]

# List of bots ID
good_accounts = tweepy.Cursor(api.followers, screen_name="sktgater").items(70)
count = 0

with open('good_accounts.csv', 'w') as table:
	writer = csv.DictWriter(table, fieldnames=attributes)
	writer.writeheader()
	for usr in good_accounts:
		try:
			dic = {attribute: getattr(usr, attribute.lower()) for attribute in attributes}
			writer.writerow(dic)
			count += 1
		except:
			continue
	print('# Of Good Accounts Written: ' + str(count))
