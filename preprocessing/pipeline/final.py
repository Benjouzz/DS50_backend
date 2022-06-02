import os
import sys
import csv
import json
import time

from util.data import read_json_rows, read_csv_rows


def process_reviews(log, user_jsontocsv:dict, reviews_in:str, reviews_out:str):
	log.title("Loading and fixing reviews")
	starttime = time.time()

	book_reviews = {}
	review_ratings = {}
	with open(reviews_out, "w", encoding="utf-8") as outfile:
		for rowindex, row in read_json_rows(reviews_in):
			user_id = user_jsontocsv[row["user_id"]]
			book_id = int(row["book_id"])
			if book_id not in book_reviews:
				book_reviews[book_id] = 1
			else:
				book_reviews[book_id] += 1

			review_ratings[(user_id, book_id)] = int(row["rating"])

			row["user_id"] = user_id
			row["book_id"] = book_id
			row["review_id"] = rowindex
			outfile.write(json.dumps(row) + "\n")

			if (rowindex + 1) % 10000 == 0:
				log.status(f"Read {len(review_ratings)} reviews for {len(book_reviews)} books")
	log.print(f"Read {len(review_ratings)} reviews for {len(book_reviews)} books")

	endtime = time.time()
	log.section(f"Section accomplished in {endtime - starttime :.3f} seconds")
	log.close()
	return book_reviews, review_ratings

def process_interactions(log, book_csvtojson:dict, review_ratings:dict, interactions_in:str, interactions_out:str):
	log.title("Loading and fixing interactions")
	starttime = time.time()

	ratings_set = 0
	non_reviewed = 0
	incoherent = 0
	book_ratings = {}
	with open(interactions_out, "w", encoding="utf-8", newline="") as outfile:
		writer = csv.DictWriter(outfile, fieldnames=["user_id", "book_id", "is_read", "rating", "is_reviewed"])
		writer.writeheader()
		for rowindex, row in read_csv_rows(interactions_in):
			row = {key: int(value) for key, value in row.items()}
			user_id = row["user_id"]
			book_id = book_csvtojson[row["book_id"]]
			key = (user_id, book_id)
			if key in review_ratings:
				rating = review_ratings[key]
				if not row["is_reviewed"]:
					row["is_reviewed"] = 1
					non_reviewed += 1
				if row["rating"] == 0:
					row["rating"] = rating
					ratings_set += 1
				elif row["rating"] != rating:
					row["rating"] = rating
					incoherent += 1

			if row["rating"] != 0:
				if book_id in book_ratings:
					book_ratings[book_id].append(row["rating"])
				else:
					book_ratings[book_id] = [row["rating"]]

			row["user_id"] = user_id
			row["book_id"] = book_id
			writer.writerow(row)


			if (rowindex + 1) % 10000 == 0:
				log.status(f"Interactions updated : {ratings_set} / {rowindex+1}, {incoherent} incoherent, {non_reviewed} given as non-reviewed")
	log.print(f"Interactions updated : {ratings_set} / {rowindex+1}, {incoherent} incoherent, {non_reviewed} given as non-reviewed")

	endtime = time.time()
	log.section(f"Section accomplished in {endtime - starttime :.3f} seconds")
	log.close()
	return book_ratings

def process_books(log, book_ratings:dict, book_reviews:dict, booktags_in:str, books_in:str, books_out:str):
	log.title("Loading and fixing books")

	log.section("Loading book tags")
	book_tags = {}
	for rowindex, row in read_json_rows(booktags_in):
		book_tags[int(row["book_id"])] = {int(key): count for key, count in row["shelves"].items()}

	log.section("Fixing books")
	starttime = time.time()

	author_books = {}
	author_ratings = {}
	author_reviews = {}
	series_books = {}
	with open(books_out, "w", encoding="utf-8") as outfile:
		for rowindex, row in read_json_rows(books_in):
			book_id = int(row["book_id"])
			ratings = book_ratings[book_id] if book_id in book_ratings else []
			reviews = book_reviews[book_id] if book_id in book_reviews else 0
			row["ratings_count"] = len(ratings)
			row["average_rating"] = sum(ratings) / len(ratings) if len(ratings) > 0 else 0
			row["text_reviews_count"] = reviews
			row["tag_count"] = len(book_tags[book_id])
			row["tag_counts_sum"] = sum(book_tags[book_id].values())

			for author in row["authors"]:
				author_id = author["author_id"]
				if author["author_id"] in author_books:
					author_books[author_id].add(book_id)
					author_ratings[author_id].extend(ratings)
					author_reviews[author_id] += reviews
				else:
					author_books[author_id] = {book_id}
					author_ratings[author_id] = ratings.copy()
					author_reviews[author_id] = reviews

			for series in row["series"]:
				if series in series_books:
					series_books[series].append(book_id)
				else:
					series_books[series] = [book_id]

			row["book_id"] = book_id
			# Just a little cleanup of unnecessary data to see more clearly
			del row["popular_shelves"]
			del row["authors"]
			del row["series"]
			del row["work_id"]

			outfile.write(json.dumps(row) + "\n")

			if (rowindex + 1) % 1000 == 0:
				log.status(f"{rowindex+1} books fixed")
	log.print(f"{rowindex+1} books fixed")

	endtime = time.time()
	log.section(f"Section accomplished in {endtime - starttime :.3f} seconds")
	log.close()
	return author_books, author_ratings, author_reviews, series_books

def process_authors(log, author_ratings:dict, author_reviews:dict, authors_in:str, authors_out:str):
	log.title("Loading and fixing authors")
	starttime = time.time()

	with open(authors_out, "w", encoding="utf-8") as outfile:
		for rowindex, row in read_json_rows(authors_in):
			author_id = row["author_id"]
			ratings = author_ratings[author_id] if author_id in author_ratings else []
			reviews = author_reviews[author_id] if author_id in author_reviews else 0

			row["ratings_count"] = len(ratings)
			row["average_rating"] = sum(ratings) / len(ratings) if len(ratings) > 0 else 0
			row["text_reviews_count"] = reviews

			outfile.write(json.dumps(row) + "\n")

			if (rowindex + 1) % 1000 == 0:
				log.status(f"{rowindex+1} authors fixed")
	log.print(f"{rowindex+1} authors fixed")

	endtime = time.time()
	log.section(f"Section accomplished in {endtime - starttime :.3f} seconds")
	log.close()

def process_wrote(log, author_books:dict, wrote_out:str):
	log.title("Writing author - book relationships")
	starttime = time.time()
	with open(wrote_out, "w", encoding="utf-8", newline="") as outfile:
		writer = csv.DictWriter(outfile, fieldnames=["author_id", "book_id"])
		writer.writeheader()
		written = 0
		for i, (author_id, books) in enumerate(author_books.items()):
			for book_id in books:
				writer.writerow({"author_id": author_id, "book_id": book_id})
				written += 1

			if (i+1) % 1000 == 0:
				log.status(f"Written {written} rows from {i+1} authors")
	log.print(f"Written {written} rows from {i+1} authors")

	endtime = time.time()
	log.section(f"Section accomplished in {endtime - starttime :.3f} seconds")
	log.close()

def process_contains(log, series_books:dict, contains_out:str):
	log.title("Writing series - book relationships")
	starttime = time.time()
	with open(contains_out, "w", encoding="utf-8", newline="") as outfile:
		writer = csv.DictWriter(outfile, fieldnames=["series_id", "book_id"])
		writer.writeheader()
		written = 0
		for i, (series_id, books) in enumerate(series_books.items()):
			for book_id in books:
				writer.writerow({"series_id": series_id, "book_id": book_id})
				written += 1

			if (i+1) % 1000 == 0:
				log.status(f"Written {written} rows in {i+1} series")
	log.print(f"Written {written} rows in {i+1} series")

	endtime = time.time()
	log.section(f"Section accomplished in {endtime - starttime :.3f} seconds")
	log.close()


if __name__ == "__main__":
	dataset_path = sys.argv[1]

	books_in = os.path.join(dataset_path, "goodreads_books.json")
	authors_in = os.path.join(dataset_path, "goodreads_book_authors.json")
	interactions_in = os.path.join(dataset_path, "goodreads_interactions.csv")
	reviews_in = os.path.join(dataset_path, "goodreads_reviews_fixed.json")
	booktags_in = os.path.join(dataset_path, "goodreads_book_tags.json")
	book_map_in = os.path.join(dataset_path, "book_id_map.csv")
	user_map_in = os.path.join(dataset_path, "user_id_map.csv")

	books_out = os.path.join(dataset_path, "goodreads_books_final.json")
	authors_out = os.path.join(dataset_path, "goodreads_book_authors_final.json")
	interactions_out = os.path.join(dataset_path, "goodreads_interactions_final.csv")
	reviews_out = os.path.join(dataset_path, "goodreads_reviews_final.json")
	wrote_out = os.path.join(dataset_path, "goodreads_wrote.csv")
	contains_out = os.path.join(dataset_path, "goodreads_series_contains.csv")

	starttime = time.time()

	print("== Loading CSV maps")
	book_jsontocsv = {}
	book_csvtojson = {}
	for rowindex, row in read_csv_rows(book_map_in):
		book_jsontocsv[int(row["book_id"])] = int(row["book_id_csv"])
		book_csvtojson[int(row["book_id_csv"])] = int(row["book_id"])

	user_jsontocsv = {}
	user_csvtojson = {}
	for rowindex, row in read_csv_rows(user_map_in):
		user_jsontocsv[row["user_id"]] = int(row["user_id_csv"])
		user_csvtojson[int(row["user_id_csv"])] = row["user_id"]

	book_reviews, review_ratings = process_reviews(user_jsontocsv, reviews_in, reviews_out)
	book_ratings = process_interactions(book_csvtojson, review_ratings, interactions_in, interactions_out)

	author_books, author_ratings, author_reviews, series_books = process_books(book_ratings, book_reviews, book_tags, books_in, books_out)
	process_authors(author_ratings, author_reviews, authors_in, authors_out)
	process_wrote(author_books, wrote_out)
	process_contains(series_books, contains_out)

	endtime = time.time()
	print(f"== Section accomplished in {endtime - starttime :.3f} seconds")