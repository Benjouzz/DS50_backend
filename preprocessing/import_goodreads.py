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
		self.cursor = None
		self.db = None

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
		if self.cursor is not None:
			self.cursor.close()
			self.cursor = None
		if self.db is not None:
			self.db.close()
			self.db = None

	def astuple(self):
		return (self.host, self.port, self.database, self.user, self.password)




class Table (object):
	"""Superclass for DB tables.
	   Needs to be subclassed once for each table"""
	
	batch_size = 2000

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

	def indexes(self):
		"""Return the additional indexes {column name: unique, ...}"""
		return {}

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

	def additional_constraints(self):
		return ""


# DB type aliases
Int = ("INTEGER", "INTEGER")
String = ("TEXT", "TINYTEXT")
Str = lambda size=None: ("TEXT", f"VARCHAR({size if size is not None else 255})")
Text = ("TEXT", "TEXT")
Float = ("FLOAT", "FLOAT")
Bool = ("BOOLEAN", "BOOLEAN")
Date = ("DATE", "DATE")
Datetime = ("DATETIME", "DATETIME")



def convert_value(value, type, emptyisnull=False):
	"""Convert a value from the base file to a python representation"""
	if type[0] == "INTEGER":
		return int(value) if value != "" else None
	elif type[0] == "FLOAT":
		return float(value) if value != "" else None
	elif type[0] == "BOOLEAN":
		return bool(value) if value != "" else None
	elif type[0] == "TEXT":
		return value if value != "" else (None if emptyisnull else value)
	elif type[0] == "DATE":
		return datetime.date.fromisoformat(value) if value != "" else None
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
	elif type[0] == "DATE":
		return "'" + value.isoformat() + "'"
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
	User = "USER"

	Wrote = "WROTE"
	Contains = "CONTAINS"
	Tagged = "TAGGED"


########## Table definitions

class BookTable (Table):
	def columns(self):
		return {
			#name                  type    pk     fk
			"book_id":            (Int,    True,  None),
			"country_code":       (Str(2), False, None),
			"description":        (Text,   False, None),
			"format":             (String, False, None),
			"image_url":          (String, False, None),
			"is_ebook":           (Bool,   False, None),
			#"language_code":     (Str(9), False, None),
			"num_pages":          (Int,    False, None),
			"publication_year":   (Int,    False, None),
			"publication_month":  (Int,    False, None),
			"publisher":          (String, False, None),
			"title":              (String, False, None),
			"average_rating":     (Float,  False, None),
			"ratings_count":      (Int,    False, None),
			"text_reviews_count": (Int,    False, None)}

	def get(self, dirname, processes):
		for rowindex, row in read_json_rows(os.path.join(dirname, "goodreads_books.json")):
			importrow = {}
			for colname, (type, pk, fk) in self._columns.items():
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
			for colname, (type, pk, fk) in self._columns.items():
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
				       {"author_id": convert_value(element["author_id"], self._columns["author_id"][0]),
				        "book_id":   convert_value(row["book_id"], self._columns["book_id"][0]),
				        "role":      convert_value(element["role"], self._columns["role"][0])})
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
			for colname, (type, pk, fk) in self._columns.items():
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
				       {"series_id": convert_value(element, self._columns["series_id"][0]),
				        "book_id":   convert_value(row["book_id"], self._columns["book_id"][0])})
				rowindex += 1

class TagTable (Table):
	def columns(self):
		return {
			"tag_id":          (Int,    True,  None),           # Guaranteed to be sequential indices, suitable for use as vector indices
			"name":            (String, False, None),
			"super":           (Int,    False, TableName.Tag),  # Super-category in the hierarchy
			"level":           (Int,    False, None),           # Depth in the hierarchy
			"favorite_select": (Bool,   False, None),           # Whether this tag is relevant to as a choice of favorite categories
		}

	def get(self, dirname, processes):
		for rowindex, row in read_csv_rows(os.path.join(dirname, "goodreads_tags.csv")):
			importrow = {}
			for colname, (type, pk, fk) in self._columns.items():
				importrow[colname] = convert_value(row[colname], type, emptyisnull=True)
			yield (rowindex, importrow)

class BookTagTable (Table):
	def columns(self):
		return {
			"book_id": (Int, True,  TableName.Book),
			"tag_id":  (Int, True,  TableName.Tag),  # Guaranteed to be sequential indices, suitable for use as vector indices
			"count":   (Int, False, None)}

	def get(self, dirname, processes):
		rowindex = 0
		for bookindex, row in read_json_rows(os.path.join(dirname, "goodreads_book_tags.json")):
			for tag_id, count in row["shelves"].items():
				importrow = {
					"book_id": convert_value(row["book_id"], self._columns["book_id"][0]),
					"tag_id":  convert_value(tag_id,         self._columns["tag_id"][0]),
					"count":   convert_value(count,          self._columns["count"][0])}
				yield (rowindex, importrow)
				rowindex += 1

class InteractionTable (Table):
	batch_size = 20000

	def columns(self):
		return {
			"user_id":     (Int,  True,  TableName.User),
			"book_id":     (Int,  True,  TableName.Book),
			"is_read":     (Bool, False, None),
			"rating":      (Int,  False, None),
			"is_reviewed": (Bool, False, None)}

	def get(self, dirname, processes):
		bookmap = {}
		for rowindex, row in read_csv_rows(os.path.join(dirname, "book_id_map.csv")):
			bookmap[int(row["book_id_csv"])] = int(row["book_id"])

		for rowindex, row in read_csv_rows(os.path.join(dirname, "goodreads_interactions.csv")):
			importrow = {}
			for colname, (type, pk, fk) in self._columns.items():
				importrow[colname] = convert_value(row[colname], type)

			importrow["book_id"] = bookmap[importrow["book_id"]]
			yield (rowindex, importrow)

class ReviewTable (Table):
	def columns(self):
		return {
			"review_id":   (Int,      True,  None),
			"user_id":     (Int,      False, TableName.User),
			"book_id":     (Int,      False, TableName.Book),
			"rating":      (Int,      False, None),
			"review_text": (Text,     False, None),
			"date_added":  (Datetime, False, None),
			"started_at":  (Datetime, False, None),
			"n_votes":     (Int,      False, None),
			"n_comments":  (Int,      False, None)}

	def get(self, dirname, processes):
		usermap = {}
		for rowindex, row in read_csv_rows(os.path.join(dirname, "user_id_map.csv")):
			usermap[row["user_id"]] = int(row["user_id_csv"])

		for rowindex, row in read_json_rows(os.path.join(dirname, "goodreads_reviews_dedup.json")):
			if row["user_id"] not in usermap:
				print(f"WARNING : Unknown user hash-ID {row['user_id']} in review {row['review_id']}. The row has been dropped.")
				continue
			else:
				importrow = {"user_id": convert_value(usermap[row["user_id"]], self._columns["user_id"][0])}

			for colname, (type, pk, fk) in self._columns.items():
				if colname not in ("user_id", "review_id"):
					importrow[colname] = convert_value(row[colname], type)
			importrow["review_id"] = rowindex
			
			yield (rowindex, importrow)

class UserTable (Table):
	def columns(self):
		return {
			"user_id": (Int, True, None),
			"first_name": (String, False, None),
			"last_name": (String, False, None),
			"username": (Str(30), False, None),
			"password": (String, False, None),
			"mail": (Str(60), False, None),
			"address": (Text, False, None),
			"sign_in_date": (Date, False, None),
			"first_fav_category": (Int, False, TableName.Tag),
			"second_fav_category": (Int, False, TableName.Tag),
			"third_fav_category": (Int, False, TableName.Tag)}

	def get(self, dirname, processes):
		for rowindex, row in read_csv_rows(os.path.join(dirname, "goodreads_users.csv")):
			importrow = {}
			for colname, (type, pk, fk) in self._columns.items():
				importrow[colname] = convert_value(row[colname], type)
			yield (rowindex, importrow)

	def additional_constraints(self):
		return "UNIQUE (username), UNIQUE (mail),\n"

	def indexes(self):
		#       colname unique
		return {"mail": True}



# Map the table names to their related table object
table_classes = {
	TableName.Book: BookTable(TableName.Book),
	TableName.Tag: TagTable(TableName.Tag),
	TableName.Tagged: BookTagTable(TableName.Tagged),
	TableName.Author: AuthorTable(TableName.Author),
	TableName.Wrote: WroteTable(TableName.Wrote),
	TableName.Series: SeriesTable(TableName.Series),
	TableName.Contains: ContainsTable(TableName.Contains),
	TableName.Interaction: InteractionTable(TableName.Interaction),
	TableName.Review: ReviewTable(TableName.Review),
	TableName.User: UserTable(TableName.User),
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

		constraints += self.table.additional_constraints()

		if constraints != "":
			constraints = constraints.rstrip("\n,")
		else:
			columns = columns.rstrip("\n,")

		self.connection.execute(f"CREATE TABLE {self.table.name} ({columns} {constraints});")

		# Generate indexes
		for colname, unique in self.table.indexes().items():
			self.connection.execute(f"CREATE {'UNIQUE ' if unique else ''}INDEX idx__{self.table.name}__{colname} ON {self.table.name}({colname});")

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
		batchkeys = set()
		self.connection.connect()
		for rowindex, row in self.table.get(dirname, processes):
			if rowindex < startrow:
				continue

			if column_names is None:
				column_names = []
				for colname, colvalue in row.items():
					column_names.append(colname)

			key = self.table.extract_key(row)
			if key in self.imported_keys or key in batchkeys:
				print(f"\rWARNING : Duplicate primary key {key} in table {self.table.name}. The row has been dropped")
				self.dropped += 1
				continue

			check_fail = False
			for colname, process in foreign_checks.items():
				if process.table.name != self.table.name and (row[colname], ) not in process.imported_keys:
					print((row[colname], ), process.imported_keys)
					print(f"\rWARNING : Foreign key {row[colname]} in column {colname} of table {self.table.name} has no target in table {process.table.name}. The row has been dropped")
					check_fail = True
			if check_fail:
				self.dropped += 1
				continue


			column_values = []
			for colname, colvalue in row.items():
				column_values.append(convert_insert(colvalue, columns[colname][0]))
			batchvalues.append("(" + ",".join(column_values) + ")")
			batchkeys.add(key)

			if len(batchvalues) >= self.table.batch_size:
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
				self.imported_keys.update(batchkeys)
				print(f"\r{self.imported} rows inserted", end="")
				batchvalues.clear()
				batchkeys.clear()

		
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
			self.imported_keys.update(batchkeys)
			batchvalues.clear()
			batchkeys.clear()

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
							for dependency in process.table.dependencies()
							if dependency != tablename]):
				order.append(tablename)
		passes += 1
		if passes > len(processes) + 1:
			print("ERROR : Endless loop in import order resolution. A circular dependency is plausible.")
			exit(5)
	return order



# For https://dbdiagram.io/d
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