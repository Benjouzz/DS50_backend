import os
import re
import csv
import sys
import json
import time
import math
import random
import shutil
import ftlangdetect


WRITTEN_PRINT_INTERVAL = 1000
READ_PRINT_INTERVAL = 50000

def roulette_sample(values, samplesize, weights):
	"""Sample individuals (without replacement) from a population with a roulette sampling
	   `values` : List of individuals
	   `samplesize` : Size of the sample to return
	   `weights` : Dict {value: weight, ...} to associate weights to each individual"""
	weights = [weights[value] for value in values]
	
	# `random.choices` is without replacement and `random.sample` does not take weights
	# So we emulate this by using `random.choices` with our weight, removing the duplicates
	# And re-filling with new random items until we’ve got our sample
	sample = set()
	while len(sample) < samplesize:
		sample |= set(random.choices(values, weights=weights, k=(samplesize-len(sample))))
	return sample


def filter_books(log, numbooks, books_in, books_out, series_in, series_out, authors_in, authors_out):
	"""SUBSTEP filter.books
	   Take a sample of books, respecting series (all books of a given series will always be present)
	   Filter the series, keeping only the sampled books’
	   Filter the authors, keeping only the sampled books’"""
	starttime = time.time()
	log.title("Filtering books and series")

	book_ids = set()
	author_ids = set()
	series_ids = set()

	book_association = {}     # Line index -> book_id
	series_dependencies = {}  # Series index -> [book_id, ...] (every book reported in the series)
	book_series = {}          # Line index -> [series_id, ...] (every series the book is in)
	book_authors = {}         # Line index -> [author_id, ...] (every author reported for the book)
	# We select books randomly based on some kind of roulette sampling with a score based on a loose definition of popularity 
	book_roulette_scores = {} # Line index -> score

	log.section("Building dependency graphs")
	with open(books_in, "r", encoding="utf-8") as books_file:
		for lineindex, line in enumerate(books_file):
			if lineindex % READ_PRINT_INTERVAL == 0:
				log.status(f"Read {lineindex} books")
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
				# Make a score based on some loose definition of popularity : ³√[ratings_count * text_reviews_count]
				# This aims at keeping a balanced dataset, instead of picking random books where there will be
				# 90% of books with only few or no interactions and reviews
				book_roulette_scores[lineindex] = (int(row["ratings_count"]) * int(row["text_reviews_count"])) ** (1/3)
				for series in row["series"]:
					if series in series_dependencies:
						series_dependencies[series].append(lineindex)
					else:
						series_dependencies[series] = [lineindex]
		
		log.section("Sampling books")
		book_count = lineindex + 1
		# Books added in the round (first round adds all sampled books)
		added_books = set(roulette_sample(list(range(book_count)), numbooks, book_roulette_scores))

		# Resolve the series dependencies, to have a stable dataset :
		# Every book has all its series and every series has all its books, such that there’s no orphan information
		log.section("Resolving series dependencies")
		added_series = set()    # Series added in the round
		included_lines = set()  # All line indices to include at the end
		numrounds = 0
		while len(added_books) > 0:  # Keep looping until there are no more changes (= stable)
			numrounds += 1
			log.write(f"Round {numrounds}")

			# Add the missing series for the books
			added_series.clear()
			for lineindex in added_books:
				included_lines.add(lineindex)
				for series_id in book_series[lineindex]:
					if series_id not in series_ids:
						series_ids.add(series_id)
						added_series.add(series_id)

			# Add the missing books in the series
			added_books.clear()
			for series_id in added_series:
				for lineindex in series_dependencies[series_id]:
					if lineindex not in included_lines:
						included_lines.add(lineindex)
						added_books.add(lineindex)

			log.print(f" : {len(added_series)} new series, {len(added_books)} new books")

		# Just write the sampled lines as they come
		log.section("Outputting filtered books")
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
						log.status(f"{written} books written")
		log.print(f"{written} books written")

	# Save with the series
	log.section("Outputting filtered series")
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
						log.status(f"{written} series written")
	log.print(f"\r{written} series written")

	# Same with the authors
	log.section("Outputting filtered authors")
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
						log.status(f"{written} authors written")
	log.print(f"{written} authors written")

	endtime = time.time()
	log.section(f"Section accomplished in {endtime-starttime :.3f} seconds")
	log.close()
	return book_ids


def filter_interactions(log, numusers, book_ids, interactions_in, interactions_out, book_map_in, book_map_out, user_map_in, user_map_out):
	"""SUBSTEP filter.interactions
	   Take a sample of users, also with a roulette sampling according to the amount of information the user created
	   Filter the interactions to keep only those for sampled books of sampled users"""
	log.title("Filtering interactions")
	starttime = time.time()
	
	# Filter the book ID mapping, only keeping the sampled books
	log.section("Filtering book ID mappings")
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
						log.status(f"{written} books mappings written")
	log.print(f"{written} books mappings written")

	# Load the interactions into memory
	log.section("Reading interactions")
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
					if written % (WRITTEN_PRINT_INTERVAL*250) == 0:
						log.status(f"{written} / {lineindex+1} interactions read")
	log.print(f"{written} / {lineindex+1} interactions read")

	# Create the user sample
	log.section(f"Sampling users ({numusers}/{len(user_interactions)})")
	# We use the (number of interactions)^2.5 as a score, to have a balanced sample instead of a majority of users with almost nothing interesting 
	user_weights = {userid: len(interactions)**(1/2.5) for userid, interactions in user_interactions.items()}
	user_sample = frozenset(roulette_sample(tuple(user_interactions.keys()), numusers, user_weights))
	output_lines = set()
	for userid in user_sample:
		output_lines.update(user_interactions[userid])

	# Finally filtering the interactions, only keeping those with the sampled books and the sampled users
	log.section("Filtering interactions")
	written = 0
	with open(interactions_in, "r", encoding="utf-8") as interactions_file:
		with open(interactions_out, "w", encoding="utf-8") as output_file:
			output_file.write(interactions_file.readline())  # Copy the header line
			for lineindex, line in enumerate(interactions_file):
				if lineindex in output_lines:
					output_file.write(line)
					written += 1
					if written % (WRITTEN_PRINT_INTERVAL*100) == 0:
						log.status(f"{written} / {lineindex+1} interactions written")
	log.print(f"\r{written} / {lineindex+1} interactions written")

	# Filter the user ID mappings, only keeping those we sampled earlier
	log.section("Filtering user ID mappings")
	written = 0
	with open(user_map_in, "r", encoding="utf-8") as usermap_file:
		with open(user_map_out, "w", encoding="utf-8") as output_file:
			output_file.write(usermap_file.readline())  # Copy the header line
			for line in usermap_file:
				csv_id, json_id = line.strip().split(",")
				if csv_id in user_sample:
					output_file.write(line)

					written += 1
					if written % (WRITTEN_PRINT_INTERVAL * 10) == 0:
						log.status(f"{written} user mappings written")
	log.print(f"{written} user mappings written")

	endtime = time.time()
	log.section(f"Section accomplished in {endtime-starttime :.3f} seconds")
	log.close()
	return user_sample


def filter_reviews(log, user_ids, user_jsontocsv, book_ids, reviews_in, reviews_out):
	"""SUBSTEP filter.reviews
	   Filter the reviews to only keep those from the sampled users on the sampled books,
	       and in English only as it’s much better for us later on"""
	log.title("Filtering reviews")
	starttime = time.time()

	written = 0
	with open(reviews_in, "r", encoding="utf-8") as reviews_file:
		with open(reviews_out, "w", encoding="utf-8") as output_file:
			for lineindex, line in enumerate(reviews_file):
				row = json.loads(line)
				#  in filtered books          and in filtered users
				if row["book_id"] in book_ids and row["user_id"] in user_jsontocsv and str(user_jsontocsv[row["user_id"]]) in user_ids:
					if ftlangdetect.detect(row["review_text"].replace("\n", " "), low_memory=False)["lang"] == "en":  # and in English
						output_file.write(line)
						written += 1
						if written % WRITTEN_PRINT_INTERVAL == 0:
							log.status(f"{written} / {lineindex+1} reviews written")
	log.print(f"{written} / {lineindex+1} reviews written")

	endtime = time.time()
	log.section(f"Section accomplished in {endtime-starttime :.3f} seconds")
	log.close()