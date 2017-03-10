import csv, tweepy

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

userlist = []

with open("bot_account_list.txt") as f:
    userlist = [line[:-1] for line in f]
print(userlist)

users = [api.get_user(username) for username in userlist]

count = 0
with open('bot_acount.csv', "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=attributes+['Bot'])
    writer.writeheader()
    for user in users:
        x = {}
        try:
            for attr in attributes:
                x[attr] = getattr(user,attr.lower()) if attr != 'Status' else user.status._json
            x['Bot'] = True
            writer.writerow(x)
            count += 1
            if count >= 50:
                break
        except:
            continue
    print(count) # number of successful stored records
