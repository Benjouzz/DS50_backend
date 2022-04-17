import os
import re
import sys
import csv
import json
import pytz
import pprint
import pickle
import argparse
import datetime
import numpy as np
import mysql.connector
from enum import Enum



class ConnectionWrapper (object):
	"""Interface of a technology-independant,
	   simple SQL database connection wrapper
	   for the simple use cases we got here"""
	def execute(self, query):
		NotImplemented

	def commit(self):
		NotImplemented

	def close(self):
		NotImplemented

class SQLiteConnectionWrapper (object):
	def __init__(self, filename):
		self.db = sqlite3.connect(filename)

	def execute(self, query):
		try:
			self.db.execute(query)
		except Exception as e:
			print("\nERROR :", e)
			print("QUERY :", query)

	def commit(self):
		self.db.commit()

	def close(self):
		self.db.close()

class MySQLConnectionWrapper (object):
	def __init__(self, host, port, database, user, password):
		self.user = user
		self.password = password
		self.host = host
		self.port = port
		self.database = database

	def connect(self):
		self.db = mysql.connector.connect(user=self.user, password=self.password, host=self.host, port=self.port, database=self.database)
		self.cursor = self.db.cursor()

	def disconnect(self):
		self.cursor.close()
		self.db.close()

	def execute(self, query):
		try:
			self.cursor.execute(query)
		except (mysql.connector.IntegrityError, mysql.connector.ProgrammingError) as e:
			print("\nERROR :", e)
			with open("faultyquery.sql", "w", encoding="utf-8") as f:
				f.write("-- " + str(e) + "\n")
				f.write(query)

	def commit(self):
		self.db.commit()
		self.cursor.close()
		self.cursor = self.db.cursor()

	def close(self):
		self.cursor.close()
		self.db.close()
		self.cursor = None
		self.db = None

	def astuple(self):
		return (self.host, self.port, self.database, self.user, self.password)




class Table (object):
	"""Superclass for DB tables.
	   Needs to be subclassed once for each table"""
	def __init__(self, name):
		self.name = name
		self._columns = self.columns()
		self._dependencies = set()
		self._primary_key = []

		# Pre-extract the primary and foreign keys from the column definitions
		for colname, (type, primary_key, foreign_key) in self._columns.items():
			if primary_key:
				self._primary_key.append(colname)
			if foreign_key is not None:
				self._dependencies.add(foreign_key)

	def columns(self):
		"""Return the columns {name: (type, pk, fk), ...}"""
		NotImplemented

	def preprocess(self, dirname, processes):
		pass

	def get(self, dirname, processes):
		"""Generate data dicts to import into the database, one for each row
		   - str dirname        : dataset root directory
		   - dict processes     : all table import process objects"""
		NotImplemented

	def dependencies(self):
		"""Return the names of the other tables this one depends onto"""
		return self._dependencies

	def primary_key(self):
		"""Return the primary key columns"""
		return self._primary_key

	def extract_key(self, row):
		"""Extract the primary key fields in the given row"""
		return tuple([row[pk_element] for pk_element in self.primary_key()])

	def add_filters(self, *filters):
		"""Add row filters to the table"""
		self.filters.extend(filters)


# DB type aliases
Int = ("INTEGER", "INTEGER")
String = ("TEXT", "TINYTEXT")
Str = lambda size=None: ("TEXT", f"VARCHAR({size if size is not None else 255})")
Text = ("TEXT", "TEXT")
Float = ("FLOAT", "FLOAT")
Bool = ("BOOLEAN", "BOOLEAN")
Datetime = ("DATETIME", "DATETIME")



def convert_value(value, type):
	"""Convert a value from the base file to a python representation"""
	if type[0] == "INTEGER":
		return int(value) if value != "" else None
	elif type[0] == "FLOAT":
		return float(value) if value != "" else None
	elif type[0] == "BOOLEAN":
		return bool(value) if value != "" else None
	elif type[0] == "TEXT":
		return value
	elif type[0] == "DATETIME":
		return datetime.datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y").astimezone(pytz.utc) if value != "" else None

def convert_insert(value, type):
	"""Convert a value to its SQL representation"""
	if value is None:
		return "NULL"
	elif type[0] in ("INTEGER", "FLOAT"):
		return str(value)
	elif type[0] == "BOOLEAN":
		return "TRUE" if value else "FALSE"
	elif type[0] == "TEXT":
		return "'" + value.replace("\\", r"\\").replace("'", r"\'").replace("\n", r"\n").replace("\t", r"\t") + "'"
	elif type[0] == "DATETIME":
		return "'" + value.strftime("%Y-%m-%d %H:%M:%S") + "'"

def read_json_rows(filename):
	with open(filename, "r", encoding="utf-8") as f:
		for i, row in enumerate(f):
			yield (i, json.loads(row))

def read_csv_rows(filename):
	with open(filename, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for i, row in enumerate(reader):
			yield (i, row)

# Table names enumeration
class TableName:
	Book = "BOOK"
	Work = "WORK"
	Author = "AUTHOR"
	Series = "SERIES"
	Interaction = "INTERACTION"
	Review = "REVIEW"
	Tag = "TAG"

	Wrote = "WROTE"
	Contains = "CONTAINS"


########## Table definitions

class BookTable (Table):
	def columns(self):
		return {
			#name                  type      pk     fk
			"book_id":            (Int,      True,  None),
			"country_code":       (Str(2),   False, None),
			"description":        (Text,     False, None),
			"format":             (String,   False, None),
			"image_url":          (String,   False, None),
			"is_ebook":           (Bool,     False, None),
			"language_code":      (Str(10),  False, None),
			"num_pages":          (Int,      False, None),
			"publication_year":   (Int,      False, None),
			"publication_month":  (Int,      False, None),
			"publisher":          (String,   False, None),
			"title":              (Str(500), False, None),
			"average_rating":     (Float,    False, None),
			"ratings_count":      (Int,      False, None),
			"text_reviews_count": (Int,      False, None)}
			#"work_id":           (Int,      False, TableName.Work)

	def get(self, dirname, processes):
		for rowindex, row in read_json_rows(os.path.join(dirname, "goodreads_books.json")):
			importrow = {}
			for colname, (type, pk, fk) in self.columns().items():
				importrow[colname] = convert_value(row[colname], type)
			yield (rowindex, importrow)


class AuthorTable (Table):
	def columns(self):
		return {
			"author_id":          (Int,    True,  None),
			"name":               (String, False, None),
			"average_rating":     (Float,  False, None),
			"ratings_count":      (Int,    False, None),
			"text_reviews_count": (Int,    False, None)}

	def get(self, dirname, processes):
		for rowindex, row in read_json_rows(os.path.join(dirname, "goodreads_book_authors.json")):
			importrow = {}
			for colname, (type, pk, fk) in self.columns().items():
				importrow[colname] = convert_value(row[colname], type)
			yield (rowindex, importrow)

class WroteTable (Table):
	def columns(self):
		return {
			"author_id": (Int,    True,  TableName.Author),
			"book_id":   (Int,    True,  TableName.Book),
			"role":      (String, False, None)}

	def get(self, dirname, processes):
		rowindex = 0
		for bookindex, row in read_json_rows(os.path.join(dirname, "goodreads_books.json")):
			importrow = {}
			for element in row["authors"]:
				yield (rowindex,
				       {"author_id": convert_value(element["author_id"], self.columns()["author_id"][0]),
				        "book_id":   convert_value(row["book_id"], self.columns()["book_id"][0]),
				        "role":      convert_value(element["role"], self.columns()["role"][0])})
				rowindex += 1

class SeriesTable (Table):
	def columns(self):
		return {
			"series_id":          (Int,    True,  None),
			"numbered":           (Bool,   False, None),
			"note":               (Text,   False, None),
			"description":        (Text,   False, None),
			"title":              (String, False, None),
			"series_works_count": (Int,    False, None),
			"primary_work_count": (Int,    False, None)}

	def get(self, dirname, processes):
		for rowindex, row in read_json_rows(os.path.join(dirname, "goodreads_book_series.json")):
			importrow = {}
			for colname, (type, pk, fk) in self.columns().items():
				importrow[colname] = convert_value(row[colname], type)
			yield (rowindex, importrow)

class ContainsTable (Table):
	def columns(self):
		return {
			"series_id": (Int, True, TableName.Series),
			"book_id":   (Int, True, TableName.Book)}

	def get(self, dirname, processes):
		rowindex = 0
		for bookindex, row in read_json_rows(os.path.join(dirname, "goodreads_books.json")):
			importrow = {}
			for element in row["series"]:
				yield (rowindex,
				       {"series_id": convert_value(element, self.columns()["series_id"][0]),
				        "book_id":   convert_value(row["book_id"], self.columns()["book_id"][0])})
				rowindex += 1

class TagTable (Table):
	# Minimum proportion of all the user tags on the book to take the tag into account
	SHELF_PROPORTION_THRESHOLD = 0.02

	# Maximum amount of operations to come from the shelf name to the checked tag name
	NAME_OPERATIONS_THRESHOLD = 3

	# Maximum proportion of difference between the shelf name and the checked tag name
	NAME_DISTANCE_THRESHOLD = 0.3

	# Minimum amount of books a tag needs to be on to be taken into account
	TAG_PRESENCE_THRESHOLD = 10

	# Maximum proportion of all the selected books a tag can be on to be taken into account
	TAG_DISCRIMINATION_THRESHOLD = 0.25

	def columns(self):
		return {
			"book_id":   (Int,     True,  TableName.Book),
			"tag_name":  (Str(25), True,  None),
			"tag_count": (Int,     False, None)}

	def preprocess(self, dirname, processes):
		bookprocess = processes[TableName.Book]

		numbooks = 0
		self.genre_books = {}
		genre_counts = {}
		association = {}
		association_counts = {}

		for bookindex, row in read_json_rows(os.path.join(dirname, "goodreads_books.json")):
			numbooks += 1
			shelves = row["popular_shelves"]
			select_threshold = sum([int(shelf["count"]) for shelf in shelves]) / (len(shelves)*0.75) if len(shelves) > 0 else 0
			for shelf in shelves:
				shelfname = shelf["name"]
				shelfcount = int(shelf["count"])
				if len(shelfname) < 25 and shelfcount > select_threshold:
					# This shelf name has already been found
					if shelfname in association:
						genre_name = association[shelfname]
						genre_counts[genre_name] += 1
						association_counts[shelfname] += 1
					else:
						# Check for a similar tag to account for typos and spelling variations (e.g "e-books" and "ebooks", "favorites" and "favourites")
						shelf_chars = set(shelfname)
						bestmatch = None
						bestdist = 10000
						for genre in genre_counts.keys():
							if abs(len(genre) - len(shelfname)) + (max(len(shelf_chars), len(set(genre))) - len(shelf_chars & set(genre))) <= self.NAME_OPERATIONS_THRESHOLD:
								operations, distance = self.distance(genre, shelfname)
								if operations <= self.NAME_OPERATIONS_THRESHOLD and distance <= self.NAME_DISTANCE_THRESHOLD and operations < bestdist:
									bestmatch = genre

						if bestmatch is not None:
							genre_name = bestmatch
							association[shelfname] = genre_name
							association_counts[shelfname] = 1
							genre_counts[genre_name] += 1
						# If everything failed, make it a new tag
						else:
							genre_name = shelfname
							association[shelfname] = genre_name
							association_counts[shelfname] = 1
							genre_counts[genre_name] = 1

					if genre_name in self.genre_books:
						if row["book_id"] in self.genre_books[genre_name]:
							self.genre_books[genre_name][row["book_id"]] += shelfcount
						else:
							self.genre_books[genre_name][row["book_id"]] = shelfcount
					else:
						self.genre_books[genre_name] = {row["book_id"]: shelfcount}
			if numbooks % 1000 == 0:
				print(f"\r{numbooks} books’ shelves processed", end="")
		print()


		# Filter the tags found by importance (not too low but also not too high, a tag present on half the books is unlikely to be of any use)
		tag_set = set()
		for genre, count in genre_counts.items():
			if count > self.TAG_PRESENCE_THRESHOLD and count < numbooks*self.TAG_DISCRIMINATION_THRESHOLD:
				# Eliminate some generic tags of type "to read", "for 2014", "currently reading", "my wishlist" or whatever
				tokens = genre.lower().replace("-", " ").replace("_", " ").split()
				if not any([token in ("read", "reading", "my", "own") or (token.isdigit() and len(token) == 4) for token in tokens]):
					tag_set.add(genre)

		# Find the associations that link to the filtered tags
		reverse_association = {}
		for key, tag in association.items():
			if tag in tag_set:
				if tag in reverse_association:
					reverse_association[tag][key] = association_counts[key]
				else:
					reverse_association[tag] = {key: association_counts[key]}

		# Select the definitive tag name for each associations group
		# We are making the hopeful guess that the majority knows how to spell
		self.final_tags = {}
		for tag, associations in reverse_association.items():
			maxkey = None
			maxcount = -1
			for key, count in associations.items():
				if count > maxcount:
					maxkey = key
					maxcount = count
			self.final_tags[tag] = maxkey


	def get(self, dirname, processes):
		rowindex = 0
		for basetag, finaltag in self.final_tags.items():
			for book_id, count in self.genre_books[basetag].items():
				yield (rowindex, {
					"book_id":   convert_value(book_id,  self.columns()["book_id"][0]),
					"tag_name":  convert_value(finaltag, self.columns()["tag_name"][0]),
					"tag_count": convert_value(count,    self.columns()["tag_count"][0])})
				rowindex += 1

	# Damerau-Levenshtein distance
	@staticmethod
	def distance(base, check):
		distance_matrix = np.zeros((len(base)+1, len(check)+1), dtype=int)
		for i in range(0, len(base)+1):
			distance_matrix[i, 0] = i
		for j in range(0, len(check)+1):
			distance_matrix[0, j] = j
		
		for i in range(1, len(base) + 1):
			for j in range(1, len(check) + 1):
				cost = (0 if base[i-1] == check[j-1] else 1)
				distance_matrix[i, j] = min(distance_matrix[i-1, j] + 1,
											distance_matrix[i, j-1] + 1,
											distance_matrix[i-1, j-1] + cost)
				if i > 1 and j > 1 and base[i-1] == check[j-2] and base[i-2] == check[j-1]:
					distance_matrix[i, j] = min(distance_matrix[i, j],
												distance_matrix[i-2, j-2] + cost)
		#return 1 - distance_matrix[len(base)-1, len(check)-1] / max(len(base), len(check))
		return distance_matrix[len(base), len(check)], distance_matrix[len(base), len(check)] / max(len(base), len(check))


class InteractionTable (Table):
	def columns(self):
		return {
			"user_id":     (Str(32), True,  None),
			"book_id":     (Int,     True,  TableName.Book),
			"is_read":     (Bool,    False, None),
			"rating":      (Int,     False, None),
			"is_reviewed": (Bool,    False, None)}

	def get(self, dirname, processes):
		usermap = {}
		for rowindex, row in read_csv_rows(os.path.join(dirname, "user_id_map.csv")):
			usermap[int(row["user_id_csv"])] = row["user_id"]

		bookmap = {}
		for rowindex, row in read_csv_rows(os.path.join(dirname, "book_id_map.csv")):
			bookmap[int(row["book_id_csv"])] = int(row["book_id"])

		for rowindex, row in read_csv_rows(os.path.join(dirname, "goodreads_interactions.csv")):
			importrow = {}
			for colname, (type, pk, fk) in self.columns().items():
				importrow[colname] = convert_value(row[colname], type)

			importrow["user_id"] = usermap[int(importrow["user_id"])]
			importrow["book_id"] = bookmap[importrow["book_id"]]
			yield (rowindex, importrow)

class ReviewTable (Table):
	def columns(self):
		return {
			"review_id":   (Str(32),  True, None),
			"user_id":     (Str(32),  False, None),
			"book_id":     (Int,      False, TableName.Book),
			"rating":      (Int,      False, None),
			"review_text": (Text,     False, None),
			"date_added":  (Datetime, False, None),
			"started_at":  (Datetime, False, None),
			"n_votes":     (Int,      False, None),
			"n_comments":  (Int,      False, None)}

	def get(self, dirname, processes):
		for rowindex, row in read_json_rows(os.path.join(dirname, "goodreads_reviews_dedup.json")):
			importrow = {}
			for colname, (type, pk, fk) in self.columns().items():
				importrow[colname] = convert_value(row[colname], type)
			yield (rowindex, importrow)



# Map the table names to their related table object
table_classes = {
	TableName.Book: BookTable(TableName.Book),
	TableName.Tag: TagTable(TableName.Tag),
	TableName.Author: AuthorTable(TableName.Author),
	TableName.Wrote: WroteTable(TableName.Wrote),
	TableName.Series: SeriesTable(TableName.Series),
	TableName.Contains: ContainsTable(TableName.Contains),
	TableName.Interaction: InteractionTable(TableName.Interaction),
	TableName.Review: ReviewTable(TableName.Review),
}


# Insertions to commit at once
BATCH_SIZE = 2000




class TableImport (object):
	"""Holds the state of a table import"""

	def __init__(self, connection, table):
		self.connection = connection
		self.table = table
		self.imported_keys = set()

	def run(self, processes, dirname, restore=False):
		"""Run the import process (table creation and data insertion)"""
		self.create()
		self.insert(processes, dirname, restore)

	def drop(self):
		self.connection.connect()
		self.connection.execute(f"DROP TABLE IF EXISTS {self.table.name};")
		self.connection.commit()
		self.connection.disconnect()

	def create(self):
		"""Drop the table if it already exists, and create it"""
		self.connection.connect()
		columns = ""
		constraints = ""

		constraints += f"PRIMARY KEY ({','.join(self.table.primary_key())}),\n"

		for colname, (type, primary_key, foreign_key) in self.table.columns().items():			
			columns += f"{colname} {type[1]},\n"

			if foreign_key is not None:
				if foreign_key not in table_classes:
					print(f"WARNING : Column {colname} in table {self.table.name} references the undefined table {foreign_key}. The foreign key constraint has been dropped")
				else:
					foreign_table = table_classes[foreign_key]
					foreign_pk = foreign_table.primary_key()
					
					if foreign_pk is None:
						print(f"WARNING : Table {foreign_key}, referenced by column {colname} in table {self.table.name}, has no primary key. The foreign key constraint has been dropped")
					elif len(foreign_pk) > 1:
						print(f"WARNING : Column {colname} in table {self.table.name} references the composite primary key of table {foreign_key}. The foreign key constraint has been dropped")
					else:
						constraints += f"FOREIGN KEY ({colname}) REFERENCES {foreign_key}({foreign_pk[0]}),\n"

		if constraints != "":
			constraints = constraints.rstrip("\n,")
		else:
			columns = columns.rstrip("\n,")

		self.connection.execute(f"CREATE TABLE {self.table.name} ({columns} {constraints});")
		self.connection.commit()
		self.connection.disconnect()

	def insert(self, processes, dirname, restore=False):
		"""Import the relevant data into the table"""
		columns = self.table.columns()
		foreign_checks = {colname: processes[tablename] for colname, (type, pk, tablename) in columns.items() if tablename is not None}

		if not restore:
			self.table.preprocess(dirname, processes)
			self.imported = 0
			self.dropped = 0
			startrow = 0
		else:
			startrow = self.imported + self.dropped

		print("Importing all valid rows")
		
		column_names = None
		batchvalues = []
		self.connection.connect()
		for rowindex, row in self.table.get(dirname, processes):
			if rowindex < startrow:
				continue

			if column_names is None:
				column_names = []
				for colname, colvalue in row.items():
					column_names.append(colname)

			key = self.table.extract_key(row)
			if key in self.imported_keys:
				print(f"\rWARNING : Duplicate primary key {key} in table {self.table.name}. The row has been dropped")
				self.dropped += 1
				continue

			check_fail = False
			for colname, process in foreign_checks.items():
				if (row[colname], ) not in process.imported_keys:
					#print(f"\rWARNING : Foreign key {row[colname]} in column {colname} of table {self.table.name} has no target in table {process.table.name}. The row has been dropped")
					check_fail = True
			if check_fail:
				self.dropped += 1
				continue


			column_values = []
			for colname, colvalue in row.items():
				column_values.append(convert_insert(colvalue, columns[colname][0]))
			batchvalues.append("(" + ",".join(column_values) + ")")

			if len(batchvalues) >= BATCH_SIZE:
				# Let’s make the hopeful guess there is nothing dangerous in our data for now
				valuestring = ',\n'.join(batchvalues)
				query = f"INSERT INTO {self.table.name} ({','.join(column_names)}) VALUES {valuestring};";
				try:
					self.connection.execute(query)
					self.connection.commit()
				except:
					save_all()
					exit(1)

				self.imported += len(batchvalues)
				print(f"\r{self.imported} rows inserted", end="")
				batchvalues.clear()

			self.imported_keys.add(key)
		
		if len(batchvalues) > 0:
			valuestring = ',\n'.join(batchvalues)
			query = f"INSERT INTO {self.table.name} ({','.join(column_names)}) VALUES {valuestring};";
			try:
				self.connection.execute(query)
				self.connection.commit()
			except:
				save_all()
				exit(1)
			self.imported += len(batchvalues)
			batchvalues.clear()

		print(f"\r{self.imported} rows inserted")
		print(f"{self.dropped} rows dropped")
		self.connection.disconnect()


def resolve_import_order(processes):
	"""Resolve the tables’ interdependencies to make a coherent import order"""
	order = []
	passes = 0
	while len(order) < len(processes):
		for tablename, process in processes.items():
			if tablename not in order and all([
							dependency in order or dependency not in table_classes
							for dependency in process.table.dependencies()]):
				order.append(tablename)
		passes += 1
		if passes > len(processes) + 1:
			print("ERROR : Endless loop in import order resolution. A circular dependency is plausible.")
			exit(5)
	return order




def generate_diagram(import_order):
	with open("diagram.txt", "w", encoding="utf-8") as er:
		for tablename in import_order:
			table = table_classes[tablename]
			foreignkeys = {}
			er.write(f"Table {tablename} {{\n")
			for colname, (type, pk, fk) in table.columns().items():
				er.write(f"\t{colname} {type[1]}{' [pk]' if pk else ''}\n")
				if fk is not None:
					foreignkeys[f"{tablename}.{colname}"] = f"{fk}.{table_classes[fk].primary_key()[0]}"
			er.write("}\n")

			for local, foreign in foreignkeys.items():
				er.write(f"Ref: {local} > {foreign}\n")


def save_all():
	print("EMERGENCY STOP")
	connection.close()
	savedata = {
		"input_directory": input_directory,
		"current_table": tablename,
		"processes": processes,
		"import_order": import_order,
		"connection_tuple": connection.astuple(),
		"table_classes": table_classes,
	}

	with open("EMERGENCY_SAVE.bak", "wb") as save:
		pickle.dump(savedata, save)
	print("EMERGENCY SAVE SUCCESSFUL, QUITTING")


########## Main script

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_directory", help="Directory where the dataset files are contained")
	parser.add_argument("--host", help="MySQL server address", default="127.0.0.1")
	parser.add_argument("--port", type=int, help="MySQL server port", default=3306)
	parser.add_argument("-u", "--user", help="MySQL database user name", default="")
	parser.add_argument("-p", "--password", help="MySQL database user password", default="")
	parser.add_argument("-d", "--database", help="Target MySQL database name")
	parser.add_argument("--diagram", action="store_true", help="Generate the ER diagram description", default=False)
	parser.add_argument("--emergencyload", action="store_true", help="Restore an emergency save", default=False)
	args = parser.parse_args()

	if args.emergencyload:
		with open("EMERGENCY_SAVE.bak", "rb") as save:
			print("STARTING FROM AN EMERGENCY SAVE")
			savedata = pickle.load(save)
			input_directory = savedata["input_directory"]
			connection = MySQLConnectionWrapper(*savedata["connection_tuple"])
			processes = savedata["processes"]
			import_order = savedata["import_order"]
			table_classes = savedata["table_classes"]
			tablename = savedata["current_table"]

			for process in processes.values():
				process.connection = connection

			print("PICKING UP INSERTION FROM TABLE", savedata["current_table"])
			current_process = processes[savedata["current_table"]]
			current_process.run(processes, input_directory, restore=True)

			for tablename in import_order[import_order.index(savedata["current_table"])+1:]:
				processes[tablename].run(processes, input_directory)
	else:
		input_directory = args.dataset_directory
		connection = MySQLConnectionWrapper(args.host, args.port, args.database, args.user, args.password)

		processes = {}
		for tablename, table in table_classes.items():
			processes[tablename] = TableImport(connection, table)

		print("\n------ Resolving import order")
		import_order = resolve_import_order(processes)
		print(f"Import order : {', '.join(import_order)}")

		if args.diagram:
			generate_diagram(import_order)
		else:
			print("\n------ Dropping existing tables")
			for tablename in reversed(import_order):
				processes[tablename].drop()

			print("\n------ Importing data")
			for tablename in import_order:
				print("--- Importing table", tablename)
				processes[tablename].run(processes, input_directory)

	connection.close()