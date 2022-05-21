import os
import re
import csv
import sys
import json
import time
import math
import random
import shutil
import langdetect


WRITTEN_PRINT_INTERVAL = 1000
READ_PRINT_INTERVAL = 10000

def roulette_sample(values, samplesize, weights):
	weights = [weights[value] for value in values]
	
	sample = set()
	while len(sample) < samplesize:
		sample |= set(random.choices(values, weights=weights, k=(samplesize-len(sample))))
	return sample


def filter_books(numbooks, books_in, books_out, series_in, series_out, authors_in, authors_out):
	starttime = time.time()
	print("\n\n======== FILTERING BOOKS AND SERIES")

	book_ids = set()
	author_ids = set()
	series_ids = set()

	book_association = {}
	series_dependencies = {}
	book_series = {}
	book_authors = {}
	# We select books randomly based on some kind of roulette sampling with a score based on a loose definition of popularity 
	book_roulette_scores = {}

	with open(books_in, "r", encoding="utf-8") as books_file:
		print("== Building dependency graphs")
		for lineindex, line in enumerate(books_file):
			if lineindex % READ_PRINT_INTERVAL == 0:
				print(f"\r{lineindex}", end="")
			row = json.loads(line)
			book_association[lineindex] = row["book_id"]
			book_series[lineindex] = row["series"]
			book_authors[lineindex] = [author["author_id"] for author in row["authors"]]

			# Eliminate some rows with null values that make them uninteresting or actively nefarious to our efforts
			# Also only keep books in english
			if (len(row["ratings_count"]) == 0 or len(row["average_rating"]) == 0 or
				len(row["is_ebook"]) == 0 or len(row["popular_shelves"]) == 0 or
				len(row["text_reviews_count"]) == 0 or not row["language_code"].startswith("en")):
				book_roulette_scores[lineindex] = 0
			else:
				# Make a score based on some loose definition of popularity : √[ratings_count * text_reviews_count]
				book_roulette_scores[lineindex] = (int(row["ratings_count"]) * int(row["text_reviews_count"])) ** (1/3)
				for series in row["series"]:
					if series in series_dependencies:
						series_dependencies[series].append(lineindex)
					else:
						series_dependencies[series] = [lineindex]
		
		print("\r== Sampling books")
		book_count = lineindex + 1
		added_books = set(roulette_sample(list(range(book_count)), numbooks, book_roulette_scores))

		print("== Resolving series dependencies")
		added_series = set()
		included_lines = set()
		numrounds = 0
		while len(added_books) > 0:
			numrounds += 1
			print(f"Round {numrounds}", end="")

			added_series.clear()
			for lineindex in added_books:
				included_lines.add(lineindex)
				for series_id in book_series[lineindex]:
					if series_id not in series_ids:
						series_ids.add(series_id)
						added_series.add(series_id)

			added_books.clear()
			for series_id in added_series:
				for lineindex in series_dependencies[series_id]:
					if lineindex not in included_lines:
						included_lines.add(lineindex)
						added_books.add(lineindex)

			print(f" : {len(added_series)} new series, {len(added_books)} new books")

		print("== Outputting filtered books")
		books_file.seek(0)
		written = 0
		with open(books_out, "w", encoding="utf-8") as output_file:
			for lineindex, line in enumerate(books_file):
				if lineindex in included_lines:
					book_ids.add(book_association[lineindex])
					output_file.write(line)

					written += 1
					if written >= len(included_lines):
						break
					if written % WRITTEN_PRINT_INTERVAL == 0:
						print(f"\r{written} books written", end="")
		print(f"\r{written} books written")

	print("== Outputting filtered series")
	written = 0
	with open(series_in, "r", encoding="utf-8") as series_file:
		with open(series_out, "w", encoding="utf-8") as output_file:
			for line in series_file:
				row = json.loads(line)
				if row["series_id"] in series_ids:
					output_file.write(line)
					written += 1
					if written >= len(series_ids):
						break
					if written % WRITTEN_PRINT_INTERVAL == 0:
						print(f"\r{written} series written", end="")
	print(f"\r{written} series written")

	print("== Outputting filtered authors")
	for lineindex in included_lines:
		for author_id in book_authors[lineindex]:
			author_ids.add(author_id)

	written = 0
	with open(authors_in, "r", encoding="utf-8") as authors_file:
		with open(authors_out, "w", encoding="utf-8") as output_file:
			for line in authors_file:
				row = json.loads(line)
				if row["author_id"] in author_ids:
					output_file.write(line)
					written += 1
					if written >= len(author_ids):
						break
					if written % WRITTEN_PRINT_INTERVAL == 0:
						print(f"\r{written} authors written", end="")
	print(f"\r{written} authors written")

	endtime = time.time()
	print(f"== Section accomplished in {endtime-starttime :.3f} seconds")
	return book_ids


def filter_interactions(numusers, book_ids, interactions_in, interactions_out, book_map_in, book_map_out, user_map_in, user_map_out):
	print("\n\n======== FILTERING INTERACTIONS")
	starttime = time.time()
	
	print("== Filtering book ID mappings")
	selected_book_csv_ids = set()
	written = 0
	with open(book_map_in, "r", encoding="utf-8") as bookmap_file:
		with open(book_map_out, "w", encoding="utf-8") as output_file:
			output_file.write(bookmap_file.readline())  # Copy the header line
			for line in bookmap_file:
				csv_id, json_id = line.strip().split(",")
				if json_id in book_ids:
					selected_book_csv_ids.add(csv_id)
					output_file.write(line)

					written += 1
					if written >= len(book_ids):
						break
					if written % WRITTEN_PRINT_INTERVAL == 0:
						print(f"\r{written} books mappings written", end="")
	print(f"\r{written} books mappings written")

	print("== Reading interactions")
	written = 0
	user_interactions = {}
	with open(interactions_in, "r", encoding="utf-8") as interactions_file:
		with open(interactions_out, "w", encoding="utf-8") as output_file:
			output_file.write(interactions_file.readline())  # Copy the header line
			for lineindex, line in enumerate(interactions_file):
				userid_csv, bookid_csv, _ = line.split(",", 2)
				if bookid_csv in selected_book_csv_ids:
					if userid_csv in user_interactions:
						user_interactions[userid_csv].add(lineindex)
					else:
						user_interactions[userid_csv] = {lineindex}
					written += 1
					if written % (WRITTEN_PRINT_INTERVAL*100) == 0:
						print(f"\r{written} / {lineindex+1} interactions read", end="")
	print(f"\r{written} / {lineindex+1} interactions read")

	print(f"== Sampling users ({numusers}/{len(user_interactions)})")
	user_sample = frozenset(roulette_sample(tuple(user_interactions.keys()), numusers, {userid: len(interactions)**(1/2.5) for userid, interactions in user_interactions.items()}))
	output_lines = set()
	for userid in user_sample:
		output_lines.update(user_interactions[userid])

	print("== Filtering interactions")
	written = 0
	with open(interactions_in, "r", encoding="utf-8") as interactions_file:
		with open(interactions_out, "w", encoding="utf-8") as output_file:
			output_file.write(interactions_file.readline())  # Copy the header line
			for lineindex, line in enumerate(interactions_file):
				if lineindex in output_lines:
					output_file.write(line)
					written += 1
					if written % (WRITTEN_PRINT_INTERVAL*10) == 0:
						print(f"\r{written} / {lineindex+1} interactions written", end="")
	print(f"\r{written} / {lineindex+1} interactions written")

	print("== Filtering user ID mappings")
	written = 0
	user_map = {}
	with open(user_map_in, "r", encoding="utf-8") as usermap_file:
		with open(user_map_out, "w", encoding="utf-8") as output_file:
			output_file.write(usermap_file.readline())  # Copy the header line
			for line in usermap_file:
				csv_id, json_id = line.strip().split(",")
				if csv_id in user_sample:
					user_map[json_id] = csv_id
					output_file.write(line)

					written += 1
					if written % WRITTEN_PRINT_INTERVAL == 0:
						print(f"\r{written} user mappings written", end="")
	print(f"\r{written} user mappings written")

	endtime = time.time()
	print(f"== Section accomplished in {endtime-starttime :.3f} seconds")

	return user_sample, user_map


def filter_reviews(user_ids, user_map, book_ids, reviews_in, reviews_out):
	print("\n\n======== FILTERING REVIEWS")
	starttime = time.time()

	written = 0
	with open(reviews_in, "r", encoding="utf-8") as reviews_file:
		with open(reviews_out, "w", encoding="utf-8") as output_file:
			for lineindex, line in enumerate(reviews_file):
				row = json.loads(line)
				try:
					if row["book_id"] in book_ids and row["user_id"] in user_map and user_map[row["user_id"]] in user_ids:
						if langdetect.detect(row["review_text"]) == "en":
							output_file.write(line)
							written += 1
							if written % WRITTEN_PRINT_INTERVAL == 0:
								print(f"\r{written} / {lineindex+1} reviews written", end="")
				except langdetect.LangDetectException:
					pass
	print(f"\r{written} / {lineindex+1} reviews written")

	endtime = time.time()
	print(f"== Section accomplished in {endtime-starttime :.3f} seconds")


if __name__ == "__main__":
	numbooks = int(sys.argv[1])
	numusers = int(sys.argv[2])
	input_dir = sys.argv[3]
	output_dir = sys.argv[4]

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	books_in = os.path.join(input_dir, "goodreads_books.json")
	books_out = os.path.join(output_dir, "goodreads_books.json")
	series_in = os.path.join(input_dir, "goodreads_book_series.json")
	series_out = os.path.join(output_dir, "goodreads_book_series.json")
	authors_in = os.path.join(input_dir, "goodreads_book_authors.json")
	authors_out = os.path.join(output_dir, "goodreads_book_authors.json")
	interactions_in = os.path.join(input_dir, "goodreads_interactions.csv")
	interactions_out = os.path.join(output_dir, "goodreads_interactions.csv")
	reviews_in = os.path.join(input_dir, "goodreads_reviews_dedup.json")
	reviews_out = os.path.join(output_dir, "goodreads_reviews_dedup.json")
	book_map_in = os.path.join(input_dir, "book_id_map.csv")
	book_map_out = os.path.join(output_dir, "book_id_map.csv")
	user_map_in = os.path.join(input_dir, "user_id_map.csv")
	user_map_out = os.path.join(output_dir, "user_id_map.csv")

	book_ids = filter_books(numbooks, books_in, books_out, series_in, series_out, authors_in, authors_out)
	user_ids, user_map = filter_interactions(numusers, book_ids, interactions_in, interactions_out, book_map_in, book_map_out, user_map_in, user_map_out)
	filter_reviews(user_ids, user_map, book_ids, reviews_in, reviews_out)