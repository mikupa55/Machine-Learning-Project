consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

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

users = tweepy.Cursor(api.followers, screen_name="mikupa55").items(100)

import csv
count = 0
with open('test.csv', "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=attributes+['Bot'])
    writer.writeheader()
    for user in users:
        try:
            x = {attr : getattr(user,attr.lower()) for attr in attributes}
            x['Bot'] = True
            writer.writerow(x)
            count += 1
        except:
            continue
    print(count)	#number of successful stored records
