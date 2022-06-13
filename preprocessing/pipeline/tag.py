import os
import sys
import json
import time
import itertools
import statistics
import numpy as np
from functools import singledispatchmethod

from nltk.corpus import wordnet
from nltk.metrics.distance import edit_distance

import process_wiktionary as wikt
from util.tag import *


NOT_DETECTED_POS_MULTIPLIER = 0.95  # Factor to apply to account for differences relative to the detected POS

#MIN_BOOKS = 4
MIN_WORD_SIMILARITY = 0.65          # Threshold underneath which a word similarity is not considered
MIN_NAME_SIMILARITY = 0.68          # Threshold beyond which a name is deemed similar enough to constitute a 1-to-1 match
MIN_NAME_SUPERCLASS_PROB = 0.76     # Threshold beyond which a name is considered a subclass of a tag

WIKTIONARY_POS_ASSOCIATION = {wordnet.NOUN: wikt.Noun, wordnet.ADJ: wikt.Adjective, wordnet.ADJ_SAT: wikt.Adjective, wordnet.VERB: wikt.Verb, wordnet.ADV: wikt.Adverb}

# Factors to apply to the similarity when moving across each relationship in the Wiktionary graph
RELATIONSHIP_SIMILARITY_FACTORS = {
	wikt.Compound:     0.82,
	wikt.Component:    0.70,
	wikt.Derived:      0.72,
	wikt.Origin:       0.72,
	wikt.Related:      0.68,
	wikt.Synonym:      0.88,
	wikt.Antonym:      0.00,
	wikt.NearSynonym:  0.80,
	wikt.NearAntonym:  0.20,
	wikt.Euphemism:    0.75,
	wikt.Hyponym:      0.74,
	wikt.Hypernym:     0.68,
	wikt.Cohyponym:    0.78,
	wikt.Holonym:      0.62,
	wikt.Meronym:      0.80,
	wikt.Comeronym:    0.71,
	wikt.Metonym:      0.70,
	wikt.Class:        0.68,
	wikt.Instance:     0.84,
	wikt.Abbreviation: 0.90,
	wikt.Expansion:    0.90,
	wikt.Alternative:  0.90,
}


# Compute the similarity based on the Damerau-Levenshtein distance between both words
def edit_similarity(shelfword:str, entryword:str):
	return 1 - (edit_distance(shelfword, entryword, transpositions=True) / max(len(shelfword), len(entryword)))

# Compute the Wu-Palmer similarity between both words in the WordNet taxonomy
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

# Recursive graph search function in the Wiktionary graph
def dig_wiktionary(wiktionary, baseentry:wikt.WiktionaryEntry, entryword:str, current_similarity:float, passed:set):
	passed.add(baseentry.id)
	if current_similarity < MIN_WORD_SIMILARITY:
		return None, None

	max_similarity = None
	max_path = None
	for relationship, ids in baseentry.relationships.items():
		for relid in ids:
			if relid in passed:
				continue

			relentry = wiktionary.entries[relid]
			passed.add(relid)
			if relentry.word == entryword:
				similarity = current_similarity * RELATIONSHIP_SIMILARITY_FACTORS[relationship]
				path = (relationship, relid)
			else:
				similarity, path = dig_wiktionary(wiktionary, relentry, entryword, current_similarity * RELATIONSHIP_SIMILARITY_FACTORS[relationship], passed)
				path = (relationship, relid) + path if path is not None else None

			if similarity is not None:
				if max_similarity is None or similarity > max_similarity:
					max_similarity = similarity
					max_path = path

	return max_similarity, max_path


# Similarity based on the graph distance in the Wiktionary graph
def wiktionary_similarity(shelfword:str, shelfpos:str, entryword:str, entrypos:str, entryhint:str, wiktionary:wikt.Wiktionary):
	entries = wiktionary.word_entries[shelfword]
	lookup_entries = []
	for entry in entries:
		if entry.pos == WIKTIONARY_POS_ASSOCIATION[shelfpos]:
			lookup_entries.append(entry)
	if len(lookup_entries) == 0:  # Probably a wrong POS detection or conversion
		lookup_entries = entries
	
	max_similarity = None
	max_path = None
	for entry in lookup_entries:
		similarity, path = dig_wiktionary(wiktionary, entry, entryword, 1.0, set())
		if similarity is not None:
			if max_similarity is None or similarity > max_similarity:
				max_similarity = similarity
				max_path = (entry.id, ) + path if path is not None else None
	return max_similarity, max_path


cache_hits = 0
cache_misses = 0
word_similarity_cache = {}

# Compute a similarity between two words, based on various methods
def word_similarity(shelfword:str, shelfpos:str, entryword:str, entrypos:str, entryhint:str, wiktionary:wikt.Wiktionary):
	global cache_hits, cache_misses
	key = (shelfword, shelfpos, entryword, entrypos, entryhint)
	if key in word_similarity_cache:
		cache_hits += 1
		return word_similarity_cache[key]

	result = wikiscore = None
	if shelfword.isdigit() != entryword.isdigit():
		#print(f"{shelfword} - {entryword} : number returns 0")
		result = 0.0
	elif shelfword.isdigit() and entryword.isdigit():
		#print(f"{shelfword} - {entryword} : number returns {result}")
		result = 1.0 if int(shelfword) == int(entryword) else 0.0
	elif shelfword == entryword:
		#print(f"{shelfword} - {entryword} : sameword returns 1")
		result = 1.0
	# At this point none are numbers
	elif shelfword in ENGLISH_WORDS and entryword in ENGLISH_WORDS and shelfpos == entrypos:
		result = wordnet_similarity(shelfword, shelfpos, entryword, entrypos, entryhint)
		#print(f"{shelfword} - {entryword} : wordnet returns 0")
	elif shelfword in wiktionary and entryword in wiktionary:
		wikiscore, wikipath = wiktionary_similarity(shelfword, shelfpos, entryword, entrypos, entryhint, wiktionary)
		result = wikiscore
	
	if result is None:
		result = edit_similarity(shelfword, entryword)

	word_similarity_cache[key] = result
	cache_misses += 1
	return result

# Compute a 1-to-1 and a many-to-1 similarity score between two names
def name_similarity(shelfname:Name, entryname:Name, wiktionary:wikt.Wiktionary):
	# Check if one is an inverted version of the other (like fiction ≠ non-fiction)
	shelf_inverters = INVERTER_WORDS & shelfname
	entry_inverters = INVERTER_WORDS & entryname
	# If the inverter words don’t match, we account for the inversion (otherwise fiction and non-fiction would be very similar)
	is_contrary = (len(shelf_inverters) % 2 != len(entry_inverters) % 2)

	shelf_words = shelfname - INVERTER_WORDS
	entry_words = entryname - INVERTER_WORDS

	# Build the similarity matrix
	similarity_matrix = np.zeros((len(shelf_words), len(entry_words)), dtype=float)
	for i, (shelfword, shelfpos, shelfhint) in enumerate(shelf_words.tokens()):
		for j, (entryword, entrypos, entryhint) in enumerate(entry_words.tokens()):
			similarity_matrix[i, j] = word_similarity(shelfword, shelfpos, entryword, entrypos, entryhint, wiktionary)

	### This is something derived from a Jaccard index (2×intersection / sum), accounting for <1 similarities and subclasses (like crime+thriller is both in categories crime and thriller)
	# Threshold
	similarity_matrix[similarity_matrix < MIN_WORD_SIMILARITY] = 0.0

	# similarity = (sum(max(lines[i])) + sum(max(columns[i]))) / (len(lines) + len(columns))
	# subclass = (sum(max(lines[i])) + sum(max(columns[i]))) / (len(lines) + len(columns where max(columns[i]) is considered))

	maxsum = 0
	subclass = 0
	subclass_denominator = 0
	for shelfindex in range(similarity_matrix.shape[0]):
		rowmax = similarity_matrix[shelfindex].max()
		if rowmax > 0:
			maxsum += rowmax
			subclass_denominator += 1
	for entryindex in range(similarity_matrix.shape[1]):
		rowmax = similarity_matrix[:, entryindex].max()
		if rowmax > 0:
			maxsum += rowmax
		subclass_denominator += 1

	similarity = maxsum / (len(shelf_words) + len(entry_words))
	if subclass_denominator > 0:
		subclass = maxsum / subclass_denominator
	else:
		subclass = 0

	if similarity > MIN_NAME_SIMILARITY and is_contrary:
		similarity = 1 - similarity
	if subclass > MIN_NAME_SIMILARITY and is_contrary:
		subclass = 0
	return similarity, subclass

# Compute the best similarity between a shelf and a tag hierarchy entry
def entry_similarity(shelfname:Name, entry:HierarchyItem, wiktionary:wikt.Wiktionary):
	max_similarity = 0
	max_subclass = 0
	for entryname in entry.names:
		similarity, subclass = name_similarity(shelfname, entryname, wiktionary)
		if (similarity > max_similarity and similarity > max_subclass) or (subclass > max_similarity and subclass > max_subclass):
			max_similarity = similarity
			max_subclass = subclass
	return max_similarity, max_subclass


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

# Categorize the user shelves into a tag hierarchy
def categorize(log, books_in:str, authors_in:str, series_in:str, tags_out:str, booktags_out:str, association_out:str, unassociated_out:str, hierarchy:Hierarchy):
	global cache_hits, cache_misses
	log.title("Tag categorization")
	log.subtitle("Loading data")

	log.section("Loading authors")
	authors = load_authors(authors_in)

	log.section("Loading series")
	series = load_series(series_in)

	starttime = time.time()

	log.section("Loading book data")
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

			if (lineindex+1) % 1000 == 0:
				log.status(f"Loaded {lineindex+1} rows, {len(existing_shelves)} shelves")
	log.print(f"Loaded {lineindex+1} rows, {len(existing_shelves)} shelves")

	"""log.subtitle("Filtering shelves")
				shelf_ids = []
				starting_shelves = len(existing_shelves)
				for shelf, nbooks in tuple(existing_shelves.items()):
					if nbooks < MIN_BOOKS:
						del existing_shelves[shelf]
					else:
						shelf_ids.append(shelf)
			
				log.print(f"{starting_shelves} -> {len(existing_shelves)} shelves")"""
	shelf_ids = list(existing_shelves.keys())

	log.section("Loading the Wiktionary data")
	wiktionary = wikt.Wiktionary.load("wiktionary.json")

	log.subtitle("Computing association")
	shelf_association = {}
	unassociated = set()
	similarity_vector = np.zeros(hierarchy.max_id+1, dtype=float)
	superclass_vector = np.zeros(hierarchy.max_id+1, dtype=float)

	for shelfindex, shelfname in enumerate(shelf_ids):
		nbooks = existing_shelves[shelfname]
		for entry in hierarchy:
			similarity, superclass_prob = entry_similarity(shelfname, entry, wiktionary)
			#print("\n", str(shelfname), str(entry.names[0]), similarity, superclass_prob)
			similarity_vector[entry.id] = similarity
			superclass_vector[entry.id] = superclass_prob

		max_similarity = np.max(similarity_vector)
		if max_similarity > MIN_NAME_SIMILARITY:  # Similar enough for a 1-to-1 match
			most_similar = [int(index) for index in np.nonzero(similarity_vector == max_similarity)[0]]
			shelf_association[shelfname] = most_similar
		else:  # Potential many-to-1 match
			superclasses = [int(index) for index in np.nonzero(superclass_vector > MIN_NAME_SUPERCLASS_PROB)[0]]
			if len(superclasses) > 0:
				shelf_association[shelfname] = superclasses
			else:
				unassociated.add(shelfname)

		if (shelfindex + 1) % 100 == 0:
			log.status(f"Processed similarity of {shelfindex+1} / {len(shelf_ids)}")
	log.print(f"Processed similarity of {shelfindex+1} / {len(shelf_ids)}")
	log.print(f"{cache_hits} cache hits, {cache_misses} cache misses")

	log.subtitle("Outputting results")
	log.section("Writing debug information")
	with open(unassociated_out, "w", encoding="utf-8") as f:
		for name in unassociated:
			f.write(name.tostring() + "\n")
	with open(association_out, "w", encoding="utf-8") as f:
		output = {}
		for shelf, associated in shelf_association.items():
			output[shelf.tostring()] = [hierarchy.get(id).names[0].tostring() for id in associated]
		json.dump(output, f, indent=4)

	log.section("Writing tag association")
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
			outfile.write(json.dumps({"book_id": str(bookid), "shelves": output_shelves}) + "\n")
			if (i+1) % 1000 == 0:
				log.status(f"Written {i+1} rows")
	log.print(f"Written {i+1} rows")

	log.section("Writing tag metadata")
	with open(tags_out, "w", encoding="utf-8") as outfile:
		outfile.write("tag_id,name,super,level,favorite_select\n")
		for entry in hierarchy:
			outfile.write(f"{entry.id},{entry.names[0]},{entry.superclass if entry.superclass is not None else ''},{entry.level},{entry.favorite_select}\n")

	endtime = time.time()
	log.section(f"Section accomplished in {endtime - starttime :.3f} seconds")
	log.close()
