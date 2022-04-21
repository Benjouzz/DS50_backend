from faker import Faker
from faker.providers import DynamicProvider
import random

import pandas as pd
import mysql.connector
from mysql.connector import Error

fake = Faker()

import hashlib

TAGS = ['fiction', 'fantasy', 'romance', 'classic', 'mystery', 'kindle', 'sci-fi', 'literature',
        'horror', 'contemporary', 'adventure', 'historical', 'adult', 'paranormal',
        'thriller', 'history', 'dystopia', 'audio', 'children', 'school', 'philosophy', 'novel', 'young'
]

category_provider = DynamicProvider(
    provider_name="favorite_category",
    elements=TAGS,
)

fake.add_provider(category_provider)

users = []

connection = mysql.connector.connect(
    host='ds50-mysql-do-user-9644544-0.b.db.ondigitalocean.com',
    database='ds50',
    user='ds50',
    password='AVNS_4ybSd0CoPKnCL5F',
    port = '25060'
)

query = f"""
SELECT
    DISTINCT user_id
FROM
    INTERACTION
;
"""

if connection.is_connected():
    users_df = pd.read_sql(query, connection)

usernames = []
mails = []

users = []

for uid in users_df['user_id']:
    already_taken = True

    while already_taken:
        user = {}
        user['user_id'] = uid
        user['first_name'] = fake.first_name()
        user['last_name'] =  fake.last_name()
        user['username'] = (user['first_name'].lower()+user['last_name'].lower()+str(random.randint(0,100)))
        user['password'] = hashlib.sha256(user['username'].encode('UTF-8')).hexdigest()
        user['mail'] = (user['first_name'][0].lower()+user['last_name'].lower()+str(random.randint(0,100))).replace(' ','')+'@'+fake.free_email_domain()
        user['address'] = fake.address()
        user['sign_in_date'] = fake.date_between()
        user['first_fav_category'] = fake.favorite_category()
        user['second_fav_category'] = fake.favorite_category()
        user['third_fav_category'] = fake.favorite_category()

        while user['second_fav_category'] in [user['first_fav_category'],user['third_fav_category']]:
            user['second_fav_category'] = fake.favorite_category()
        while user['third_fav_category'] in [user['first_fav_category'],user['second_fav_category']]:
            user['third_fav_category'] = fake.favorite_category()

        if user['username'] in usernames or user['mail'] in mails:
            already_take = True
        else:
            already_taken = False
            usernames.append(user['username'])
            mails.append(user['mail'])
            users.append(user)