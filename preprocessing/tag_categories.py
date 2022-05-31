import os
import sys
import json
import pprint
import itertools
import statistics
import numpy as np
from functools import singledispatchmethod

from nltk.corpus import wordnet
from nltk.metrics.distance import edit_distance

import process_wiktionary as wikt
from tag_util import *


wiktionary = []  #wikt.Wiktionary.load("wiktionary.json")

NOT_DETECTED_POS_MULTIPLIER = 0.95

MIN_BOOKS = 4
MIN_WORD_SIMILARITY = 0.65
MIN_SAME_SIMILARITY = 0.70
MIN_NAME_SIMILARITY = 0.72
MIN_NAME_SUPERCLASS_PROB = 0.78


def edit_similarity(shelfword:str, entryword:str):
	return 1 - (edit_distance(shelfword, entryword, transpositions=True) / max(len(shelfword), len(entryword)))

def wordnet_similarity(shelfword:str, shelfpos:str, entryword:str, entrypos:str, entryhint:str):
	entry_synsets = wordnet.synsets(entryword, entrypos) if entryhint is None else [wordnet.synset(entryhint)]
	shelf_synsets = []
	for synset in wordnet.synsets(shelfword):
		if synset.pos == entrypos or (synset.pos == "a" and entrypos == "s") or (synset.pos == "s" and entrypos == "a"):
			shelf_synsets.append((synset, synset.pos == shelfpos or (synset.pos == "s" and shelfpos == "a")))

	max_similarity = 0.0
	for (shelf_synset, has_detected_pos), entry_synset in itertools.product(shelf_synsets, entry_synsets):
		similarity = wordnet.wup_similarity(shelf_synset, entry_synset) * (NOT_DETECTED_POS_MULTIPLIER*(not has_detected_pos))
		max_similarity = max(max_similarity, similarity)

	return max_similarity


def wiktionary_similarity(shelfword:str, shelfpos:str, entryword:str, entrypos:str, entryhint:str):
	return 0.0


def word_similarity(shelfword:str, shelfpos:str, entryword:str, entrypos:str, entryhint:str):
	if shelfword.isdigit() != entryword.isdigit():
		return 0.0
	elif shelfword.isdigit() and entryword.isdigit():
		return 1.0 if int(shelfword) == int(entryword) else 0.0
	elif shelfword == entryword:
		return 1.0
	# At this point none are numbers
	elif shelfword in ENGLISH_WORDS and entryword in ENGLISH_WORDS and shelfpos == entrypos:
		return wordnet_similarity(shelfword, shelfpos, entryword, entrypos, entryhint)
	elif shelfword in wiktionary and entryword in wiktionary:
		return wiktionary_similarity(shelfword, shelfpos, entryword, entrypos, entryhint)
	else:
		return edit_similarity(shelfword, entryword)

def name_similarity(shelfname:Name, entryname:Name):
	shelf_inverters = INVERTER_WORDS & shelfname
	entry_inverters = INVERTER_WORDS & entryname
	is_contrary = (len(shelf_inverters) % 2 != len(entry_inverters) % 2)

	shelf_words = shelfname - INVERTER_WORDS
	entry_words = entryname - INVERTER_WORDS
	

	similarity_matrix = np.zeros((len(shelf_words), len(entry_words)), dtype=float)
	for i, (shelfword, shelfpos, shelfhint) in enumerate(shelf_words.tokens()):
		for j, (entryword, entrypos, entryhint) in enumerate(entry_words.tokens()):
			similarity_matrix[i, j] = word_similarity(shelfword, shelfpos, entryword, entrypos, entryhint)

	# TODO : Find something better
	similarity_matrix[similarity_matrix < MIN_WORD_SIMILARITY] = 0.0

	similarity = 0
	subclass = 0
	subclass_denominator = 0
	for shelfindex in range(similarity_matrix.shape[0]):
		rowmax = similarity_matrix[shelfindex].max()
		if rowmax > 0:
			similarity += rowmax
			subclass += rowmax
			subclass_denominator += 1
	for entryindex in range(similarity_matrix.shape[1]):
		similarity += similarity_matrix[:, entryindex].max()

	similarity /= (len(shelf_words) + len(entry_words))
	if subclass_denominator > 0:
		subclass /= subclass_denominator
	else:
		subclass = 0

	if similarity > MIN_SAME_SIMILARITY and is_contrary:
		similarity = 1 - similarity
	if subclass > MIN_SAME_SIMILARITY and is_contrary:
		subclass = 0
	return similarity, subclass

def entry_similarity(shelfname:Name, entry:HierarchyItem):
	max_similarity = 0
	max_subclass = 0
	for entryname in entry.names:
		similarity, subclass = name_similarity(shelfname, entryname)
		if (similarity > max_similarity and similarity > max_subclass) or (subclass > max_similarity and subclass > max_subclass):
			max_similarity = similarity
			max_subclass = subclass
	return max_similarity, max_subclass


def categorize(books_in:str, tags_out:str, booktags_out:str, association_out:str, unassociated_out:str, hierarchy:Hierarchy, authors:dict, series:dict):
	print("\n==== Loading book data")
	book_shelves = {}
	existing_shelves = {}
	with open(books_in, "r", encoding="utf-8") as bookfile:
		for lineindex, line in enumerate(bookfile):
			row = json.loads(line)
			if len(row["popular_shelves"]) == 0:
				continue

			bookid = int(row["book_id"])
			book_shelves[bookid] = {}
			threshold = statistics.harmonic_mean([int(shelf["count"]) for shelf in row["popular_shelves"]])  # Using harmonic mean here because what we are doing looks a little bit like an Fst. TODO : check if it’s right
			for shelf in row["popular_shelves"]:
				count = int(shelf["count"])
				if count < threshold or not is_latin(shelf["name"]):
					continue

				name = Name(shelf["name"])
				if (len(name) == 0 or  # Name conversion eliminated all words
						len(INITIAL_BANWORDS & name) > 0 or  # Contains a banword
						any({len(authors[author["author_id"]] & name) > 0 for author in row["authors"]}) or  # Tag that denotes the author, redundant
						name in {series[id] for id in row["series"]}):  # Tag that denotes the series, redundant
					continue

				book_shelves[bookid][name] = count
				if name in existing_shelves:
					existing_shelves[name] += 1
				else:
					existing_shelves[name] = 1

			if (lineindex+1) % 100 == 0:
				print(f"\rLoaded {lineindex+1} rows, {len(existing_shelves)} shelves", end="")
	print(f"\rLoaded {lineindex+1} rows, {len(existing_shelves)} shelves")

	print("\n==== Filtering shelves")
	shelf_ids = []
	starting_shelves = len(existing_shelves)
	for shelf, nbooks in tuple(existing_shelves.items()):
		if nbooks < MIN_BOOKS:
			del existing_shelves[shelf]
		else:
			shelf_ids.append(shelf)

	pprint.pprint(existing_shelves)
	print(f"{starting_shelves} -> {len(existing_shelves)} shelves")

	print("\n==== Computing association")
	shelf_association = {}
	unassociated = set()
	similarity_vector = np.zeros(hierarchy.max_id+1, dtype=float)
	superclass_vector = np.zeros(hierarchy.max_id+1, dtype=float)

	for shelfindex, shelfname in enumerate(shelf_ids):
		nbooks = existing_shelves[shelfname]
		for entry in hierarchy:
			similarity, superclass_prob = entry_similarity(shelfname, entry)
			#print("\n", str(shelfname), str(entry.names[0]), similarity, superclass_prob)
			similarity_vector[entry.id] = similarity
			superclass_vector[entry.id] = superclass_prob

		max_similarity = np.max(similarity_vector)
		if max_similarity > MIN_NAME_SIMILARITY:
			most_similar = [int(index) for index in np.nonzero(similarity_vector == max_similarity)[0]]
			shelf_association[shelfname] = most_similar
		else:
			superclasses = [int(index) for index in np.nonzero(superclass_vector > MIN_NAME_SUPERCLASS_PROB)[0]]
			if len(superclasses) > 0:
				shelf_association[shelfname] = superclasses
			else:
				unassociated.add(shelfname)

		print(f"\rProcessed similarity of {shelfindex+1}/{len(shelf_ids)}", end="")
	print(f"\rProcessed similarity of {shelfindex+1}/{len(shelf_ids)}")

	print("==== Writing debug information")
	with open(unassociated_out, "w", encoding="utf-8") as f:
		for name in unassociated:
			f.write(name.tostring() + "\n")
	with open(association_out, "w", encoding="utf-8") as f:
		output = {}
		for shelf, associated in shelf_association.items():
			output[shelf.tostring()] = [hierarchy.get(id).names[0].tostring() for id in associated]
		json.dump(output, f, indent=4)

	print("==== Writing tag association")
	with open(booktags_out, "w", encoding="utf-8") as outfile:
		for i, (bookid, shelves) in enumerate(book_shelves.items()):
			output_shelves = {}
			for shelfname, count in shelves.items():
				if shelfname in shelf_association:
					for category_id in shelf_association[shelfname]:
						if category_id in output_shelves:
							output_shelves[category_id] += count
						else:
							output_shelves[category_id] = count
			outfile.write(json.dumps({"book_id": bookid, "shelves": output_shelves}) + "\n")
			if (i+1) % 100 == 0:
				print(f"\rWritten {i+1} rows", end="")
	print(f"\rWritten {i+1} rows")

	print("==== Writing tag metadata")
	with open(tags_out, "w", encoding="utf-8") as outfile:
		outfile.write("tag_id,name,super,level,favorite_select\n")
		for entry in hierarchy:
			outfile.write(f"{entry.id},{entry.names[0]},{entry.superclass if entry.superclass is not None else ''},{entry.level},{entry.favorite_select}\n")




def load_authors(authors_in:str):
	authors = {}
	with open(authors_in, "r", encoding="utf-8") as authorfile:
		for i, line in enumerate(authorfile):
			row = json.loads(line)
			authors[row["author_id"]] = Name(row["name"])
	return authors

def load_series(series_in:str):
	series = {}
	with open(series_in, "r", encoding="utf-8") as seriesfile:
		for i, line in enumerate(seriesfile):
			row = json.loads(line)
			series[row["series_id"]] = Name(row["title"])
	return series

if __name__ == "__main__":
	dataset_path = sys.argv[1]
	books_in = os.path.join(dataset_path, "goodreads_books.json")
	authors_in = os.path.join(dataset_path, "goodreads_book_authors.json")
	series_in = os.path.join(dataset_path, "goodreads_book_series.json")
	booktags_out = os.path.join(dataset_path, "goodreads_book_tags.json")
	tags_out = os.path.join(dataset_path, "goodreads_tags.csv")
	association_out = os.path.join(dataset_path, "association.json")
	unassociated_out = os.path.join(dataset_path, "unassociated.txt")

	print("== Loading authors and series")
	authors = load_authors(authors_in)
	series = load_series(series_in)

	with open("category-hierarchy.json", "r", encoding="utf-8") as jsonfile:
		hierarchy = Hierarchy.load(json.load(jsonfile))

	categorize(books_in, tags_out, booktags_out, association_out, unassociated_out, hierarchy, authors, series)