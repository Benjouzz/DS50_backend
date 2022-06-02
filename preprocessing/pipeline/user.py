import os
import sys
import csv
import json
import time
import random
import hashlib
from faker import Faker
from util.tag import Hierarchy
from util.data import read_csv_rows, read_json_rows

def random_variation():
	return random.random() / 2 + 0.75

def generate_users(log, user_csvtojson:dict, book_csvtojson:dict, interactions_in:str, booktags_in:str, users_out:str, hierarchy:Hierarchy):
	log.title("Generating fake users")
	log.subtitle("Loading data")

	log.section("Loading tags")
	tagmap = {}
	tagsums = {}
	existing_tags = set()
	for rowindex, row in read_json_rows(booktags_in):
		tagsum = sum([count for tag_id, count in row["shelves"].items()])
		tagmap[int(row["book_id"])] = {int(tag_id): count / tagsum for tag_id, count in row["shelves"].items()}
		existing_tags.update({int(tag_id) for tag_id, count in row["shelves"].items()})
	existing_tags = tuple(existing_tags)

	log.section("Loading interactions")
	interactions = {}
	for rowindex, row in read_csv_rows(interactions_in):
		user_id = int(row["user_id"])
		if user_id in interactions:
			interactions[user_id].append(row)
		else:
			interactions[user_id] = [row]

	log.subtitle("Generating users")
	starttime = time.time()

	usernames = set()
	fake = Faker()
	random_favorites = 0
	with open(users_out, "w", encoding="utf-8", newline="") as out:
		writer = csv.DictWriter(out, fieldnames=["user_id", "first_name", "last_name", "username", "password", "mail", "address", "sign_in_date", "first_fav_category", "second_fav_category", "third_fav_category"])
		writer.writeheader()
		for i, user_id in enumerate(user_csvtojson.keys()):
			first_name = fake.first_name()
			last_name = fake.last_name()
			
			username_base = (first_name[0].lower() + last_name.lower()).replace(" ", "")
			username_num = 1
			username = username_base
			while username in usernames:
				username = username_base + str(username_num)
				username_num += 1
			usernames.add(username)

			password = hashlib.sha256(username.encode("UTF-8")).hexdigest()
			email = username + "@" + fake.free_email_domain()
			address = fake.address().replace("\n", ", ")
			signin_date = fake.date_between()

			if user_id in interactions:
				user_books = set()
				for interaction in interactions[user_id]:
					user_books.add(book_csvtojson[int(interaction["book_id"])])

				user_tags = {}
				for book_id in user_books:
					if book_id in tagmap:
						for tag_id, proportion in tagmap[book_id].items():
							if hierarchy.get(tag_id).favorite_select:
								if tag_id in user_tags:
									user_tags[tag_id] += proportion
								else:
									user_tags[tag_id] = proportion
				favorites = [tag for tag, score in sorted(user_tags.items(), key=lambda item: -item[1]*random_variation())[:3]]
			else:
				favorites = []
			if len(favorites) < 3:
				num_favorites = 3 - len(favorites)
				favorites.extend(random.sample(existing_tags, num_favorites))
				random_favorites += num_favorites

			row = {
				"user_id": user_id,
				"first_name": first_name,
				"username": username,
				"password": password,
				"mail": email,
				"address": address,
				"sign_in_date": signin_date,
				"first_fav_category": favorites[0],
				"second_fav_category": favorites[1],
				"third_fav_category": favorites[2],
			}
			writer.writerow(row)
			
			if (i+1) % 2000 == 0:
				log.status(f"Users generated : {i+1} / {len(user_csvtojson)}, {random_favorites} random interactions")
	log.print(f"Users generated : {i+1} / {len(user_csvtojson)}, {random_favorites} random interactions")

	endtime = time.time()
	log.section(f"Section accomplished in {endtime - starttime :.3f} seconds")
	log.close()