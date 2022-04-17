import os
import re
import csv
import sys
import json
import time
import random
import shutil


WRITTEN_PRINT_INTERVAL = 1000
READ_PRINT_INTERVAL = 10000


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

	with open(books_in, "r", encoding="utf-8") as books_file:
		print("== Building dependency graphs")
		for lineindex, line in enumerate(books_file):
			if lineindex % READ_PRINT_INTERVAL == 0:
				print(f"\r{lineindex}", end="")
			row = json.loads(line)
			book_association[lineindex] = row["book_id"]
			book_series[lineindex] = row["series"]
			book_authors[lineindex] = [author["author_id"] for author in row["authors"]]
			for series in row["series"]:
				if series in series_dependencies:
					series_dependencies[series].append(lineindex)
				else:
					series_dependencies[series] = [lineindex]
		
		print("\r== Resolving series dependencies")
		book_count = lineindex + 1
		added_books = set(random.sample(list(range(book_count)), numbooks))
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


def filter_interactions(book_ids, interactions_in, interactions_out, book_map_in, book_map_out, user_map_in, user_map_out):
	print("\n\n======== FILTERING INTERACTIONS")
	starttime = time.time()
	
	print("== Filtering book ID mappings")
	selected_csv_ids = set()
	written = 0
	with open(book_map_in, "r", encoding="utf-8") as bookmap_file:
		with open(book_map_out, "w", encoding="utf-8") as output_file:
			output_file.write(bookmap_file.readline())  # Copy the header line
			for line in bookmap_file:
				csv_id, json_id = line.strip().split(",")
				if json_id in book_ids:
					selected_csv_ids.add(csv_id)
					output_file.write(line)

					written += 1
					if written >= len(book_ids):
						break
					if written % WRITTEN_PRINT_INTERVAL == 0:
						print(f"\r{written} books mappings written", end="")
	print(f"\r{written} books mappings written")

	print("== Copying user ID mappings")
	shutil.copy(user_map_in, user_map_out)

	print("== Filtering interactions")
	written = 0
	with open(interactions_in, "r", encoding="utf-8") as interactions_file:
		with open(interactions_out, "w", encoding="utf-8") as output_file:
			output_file.write(interactions_file.readline())  # Copy the header line
			for lineindex, line in enumerate(interactions_file):
				userid_csv, bookid_csv, _ = line.split(",", 2)
				if bookid_csv in selected_csv_ids:
					output_file.write(line)
					written += 1
					if written % (WRITTEN_PRINT_INTERVAL*5) == 0:
						print(f"\r{written} / {lineindex+1} interactions written", end="")
	print(f"\r{written} / {lineindex+1} interactions written")

	endtime = time.time()
	print(f"== Section accomplished in {endtime-starttime :.3f} seconds")


def filter_reviews(book_ids, reviews_in, reviews_out):
	print("\n\n======== FILTERING REVIEWS")
	starttime = time.time()

	written = 0
	with open(reviews_in, "r", encoding="utf-8") as reviews_file:
		with open(reviews_out, "w", encoding="utf-8") as output_file:
			for lineindex, line in enumerate(reviews_file):
				row = json.loads(line)
				if row["book_id"] in book_ids:
					output_file.write(line)
					written += 1
					if written % (WRITTEN_PRINT_INTERVAL*2) == 0:
						print(f"\r{written} / {lineindex+1} reviews written", end="")
	print(f"\r{written} / {lineindex+1} reviews written")

	endtime = time.time()
	print(f"== Section accomplished in {endtime-starttime :.3f} seconds")


if __name__ == "__main__":
	numbooks = int(sys.argv[1])
	input_dir = sys.argv[2]
	output_dir = sys.argv[3]

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
	filter_interactions(book_ids, interactions_in, interactions_out, book_map_in, book_map_out, user_map_in, user_map_out)
	filter_reviews(book_ids, reviews_in, reviews_out)