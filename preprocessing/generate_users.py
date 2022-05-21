import os
import sys
import csv
import json
import random
import hashlib
from faker import Faker
from tag_util import Hierarchy


def read_json_rows(filename):
	with open(filename, "r", encoding="utf-8") as f:
		for i, row in enumerate(f):
			yield (i, json.loads(row))

def read_csv_rows(filename):
	with open(filename, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for i, row in enumerate(reader):
			yield (i, row)

def random_variation():
	return random.random() / 2 + 0.75

def generate_users(usermap_in:str, bookmap_in:str, interaction_in:str, booktags_in:str, users_out:str):
	print("== Loading data")
	tagmap = {}
	tagsums = {}
	existing_tags = set()
	for rowindex, row in read_json_rows(booktags_in):
		tagsum = sum([count for tag_id, count in row["shelves"].items()])
		tagmap[row["book_id"]] = {int(tag_id): count / tagsum for tag_id, count in row["shelves"].items()}
		existing_tags.update({int(tag_id) for tag_id, count in row["shelves"].items()})
	existing_tags = tuple(existing_tags)

	interactions = {}
	for rowindex, row in read_csv_rows(interactions_in):
		user_id = int(row["user_id"])
		if user_id in interactions:
			interactions[user_id].append(row)
		else:
			interactions[user_id] = [row]

	usermap = set()
	for rowindex, row in read_csv_rows(usermap_in):
		usermap.add(int(row["user_id_csv"]))

	bookmap = {}
	for rowindex, row in read_csv_rows(bookmap_in):
		bookmap[int(row["book_id_csv"])] = int(row["book_id"])

	with open("category-hierarchy.json", "r", encoding="utf-8") as f:
		hierarchy = Hierarchy.load(json.load(f))

	print("== Generating users")
	usernames = set()
	fake = Faker()
	random_favorites = 0
	with open(users_out, "w", encoding="utf-8") as out:
		out.write("user_id,first_name,last_name,username,password,mail,address,sign_in_date,first_fav_category,second_fav_category,third_fav_category\n")
		for i, user_id in enumerate(usermap):
			first_name = fake.first_name()
			last_name = fake.last_name()
			
			username_base = first_name[0].lower() + last_name.lower()
			username_num = 1
			username = username_base
			while username in usernames:
				username = username_base + str(username_num)
				username_num += 1

			password = hashlib.sha256(username.encode("UTF-8")).hexdigest()
			email = username.replace(" ", "") + "@" + fake.free_email_domain()
			address = fake.address().replace("\n", ", ")
			signin_date = fake.date_between()

			if user_id in interactions:
				user_books = set()
				for interaction in interactions[user_id]:
					user_books.add(bookmap[int(interaction["book_id"])])

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
			out.write(f"{user_id},{first_name},{last_name},{username},{password},{email},\"{address}\",{signin_date},{favorites[0]},{favorites[1]},{favorites[2]}\n")

			if (i+1) % 500 == 0:
				print(f"\rUsers generated : {i+1} / {len(usermap)}, {random_favorites} random interactions", end="")
	print(f"\rUsers generated : {i+1} / {len(usermap)}, {random_favorites} random interactions")



if __name__ == "__main__":
	dataset_path = sys.argv[1]
	interactions_in = os.path.join(dataset_path, "goodreads_interactions.csv")
	booktags_in = os.path.join(dataset_path, "goodreads_book_tags.json")
	bookmap_in = os.path.join(dataset_path, "book_id_map.csv")
	usermap_in = os.path.join(dataset_path, "user_id_map.csv")
	users_out = os.path.join(dataset_path, "goodreads_users.csv")

	generate_users(usermap_in, bookmap_in, interactions_in, booktags_in, users_out)