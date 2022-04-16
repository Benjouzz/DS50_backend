import os
import sys
import csv
import json
import pytz
import argparse
import datetime
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
		self.db = mysql.connector.connect(user=user, password=password, host=host, port=port, database=database)
		self.cursor = self.db.cursor()

	def execute(self, query):
		try:
			self.cursor.execute(query)
		except Exception as e:
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




class Table (object):
	"""Superclass for DB tables.
	   Needs to be subclassed once for each table"""
	def __init__(self, name):
		self.name = name
		self.truncation = None
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

	def get(self, dirname, processes, selectedrows=None):
		"""Generate data dicts to import into the database, one for each row
		   - str dirname        : dataset root directory
		   - dict processes     : all table import process objects
		   - int[] selectedrows : row indices to sample (all others will be skipped)"""
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

	def truncate(self, truncation):
		"""Set a limitation to the import : only the first `truncation` rows will be considered"""
		self.truncation = truncation


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

def read_json_rows(filename, selectedrows=None, truncate=None):
	"""Decode JSON rows from the given file
	   - str filename       : file to read
	   - int[] selectedrows : row indices to keep (if left to None, select all)
	   - int truncate       : limit on the number of rows to read (if left to None, select all)"""
	with open(filename, "r", encoding="utf-8") as f:
		for i, row in enumerate(f):
			if truncate is not None and i >= truncate:
				break
			if selectedrows is None or i in selectedrows:
				yield (i, json.loads(row))

def read_csv_rows(filename, selectedrows=None, truncate=None):
	"""Decode CSV rows from the given file
	   - str filename       : file to read
	   - int[] selectedrows : row indices to keep (if left to None, select all)
	   - int truncate       : limit on the number of rows to read (if left to None, select all)"""
	with open(filename, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for i, row in enumerate(reader):
			if truncate is not None and i >= truncate:
				break
			if selectedrows is None or i in selectedrows:
				yield (i, row)

# Table names enumeration
class TableName:
	Book = "BOOK"
	Work = "WORK"
	Author = "AUTHOR"
	Interaction = "INTERACTION"
	Review = "REVIEW"

	Wrote = "WROTE"


########## Table definitions

class BookTable (Table):
	def columns(self):
		return {
			#name                  type      pk     fk
			"book_id":            (Int,      True,  None),
			"country_code":       (Str(2),   False, None),
			#"description":       (Text,     False, None),
			"format":             (String,   False, None),
			"image_url":          (String,   False, None),
			"is_ebook":           (Bool,     False, None),
			"language_code":      (Str(5),   False, None),
			"num_pages":          (Int,      False, None),
			"publication_year":   (Int,      False, None),
			"publication_month":  (Int,      False, None),
			"publisher":          (String,   False, None),
			"title":              (Str(500), False, None),
			"average_rating":     (Float,    False, None),
			"ratings_count":      (Int,      False, None),
			"text_reviews_count": (Int,      False, None)}
			#"work_id":           (String, False, TableName.Work)

	def get(self, dirname, processes, selectedrows=None):
		for rowindex, row in read_json_rows(os.path.join(dirname, "goodreads_books.json"), selectedrows, self.truncation):
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

	def get(self, dirname, processes, selectedrows=None):
		for rowindex, row in read_json_rows(os.path.join(dirname, "goodreads_book_authors.json"), selectedrows, self.truncation):
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

	def get(self, dirname, processes, selectedrows=None):
		bookprocess = processes[TableName.Book]

		rowindex = 0
		for bookindex, row in read_json_rows(os.path.join(dirname, "goodreads_books.json"), bookprocess.rows_to_import, self.truncation):
			importrow = {}
			for element in row["authors"]:
				if selectedrows is None or rowindex in selectedrows:
					yield (rowindex,
					       {"author_id": convert_value(element["author_id"], self.columns()["author_id"][0]),
					        "book_id":   convert_value(row["book_id"], self.columns()["book_id"][0]),
					        "role":      convert_value(element["role"], self.columns()["role"][0])})
				rowindex += 1

class InteractionTable (Table):
	def columns(self):
		return {
			"user_id":     (Str(32), True,  None),
			"book_id":     (Int,     True,  TableName.Book),
			"is_read":     (Bool,    False, None),
			"rating":      (Int,     False, None),
			"is_reviewed": (Bool,    False, None)}

	def get(self, dirname, processes, selectedrows=None):
		usermap = {}
		for rowindex, row in read_csv_rows(os.path.join(dirname, "user_id_map.csv")):
			usermap[int(row["user_id_csv"])] = row["user_id"]

		bookmap = {}
		for rowindex, row in read_csv_rows(os.path.join(dirname, "book_id_map.csv")):
			bookmap[int(row["book_id_csv"])] = int(row["book_id"])

		for rowindex, row in read_csv_rows(os.path.join(dirname, "goodreads_interactions.csv"), selectedrows, self.truncation):
			importrow = {}
			for colname, (type, pk, fk) in self.columns().items():
				importrow[colname] = convert_value(row[colname], type)

			importrow["user_id"] = usermap[int(importrow["user_id"])]
			importrow["book_id"] = usermap[importrow["book_id"]]
			yield (rowindex, importrow)

class ReviewTable (Table):
	def columns(self):
		return {
			"review_id":   (Str(32),  True, None),
			"user_id":     (Str(32),  False, None),
			"book_id":     (Int,      False, TableName.Book),
			"rating":      (Int,      False, None),
			#"review_text": (Text,     False, None),
			"date_added":  (Datetime, False, None),
			"started_at":  (Datetime, False, None),
			"n_votes":     (Int,      False, None),
			"n_comments":  (Int,      False, None)}

	def get(self, dirname, processes, selectedrows=None):
		for rowindex, row in read_json_rows(os.path.join(dirname, "goodreads_reviews_dedup.json"), selectedrows, self.truncation):
			importrow = {}
			for colname, (type, pk, fk) in self.columns().items():
				importrow[colname] = convert_value(row[colname], type)
			yield (rowindex, importrow)



# Map the table names to their related table object
table_classes = {
	TableName.Book: BookTable(TableName.Book),
	TableName.Wrote: WroteTable(TableName.Wrote),
	TableName.Author: AuthorTable(TableName.Author),
	TableName.Interaction: InteractionTable(TableName.Interaction),
#	TableName.Review: ReviewTable(TableName.Review),
}


# Insertions to commit at once
BATCH_SIZE = 1000




class TableImport (object):
	"""Holds the state of a table import"""

	def __init__(self, connection, table):
		self.connection = connection
		self.table = table
		self.rows_to_import = None
		self.imported_keys = set()

		if self.table.truncation is not None:
			self.rows_to_import = set(range(self.table.truncation))

	def run(self, processes, dirname):
		"""Run the import process (table creation and data insertion)"""
		self.create()
		self.insert(processes, dirname)

	def drop(self):
		connection.execute(f"DROP TABLE IF EXISTS {self.table.name};")
		connection.commit()

	def create(self):
		"""Drop the table if it already exists, and create it"""
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

		connection.execute(f"CREATE TABLE {self.table.name} ({columns} {constraints});")
		connection.commit()

	def insert(self, processes, dirname):
		"""Import the relevant data into the table"""
		columns = self.table.columns()
		imported = 0

		if self.rows_to_import is None:
			print("Importing all valid rows")
		else:
			print(f"{len(self.rows_to_import)} rows to import")

		foreign_checks = {colname: processes[tablename] for colname, (type, pk, tablename) in self.table.columns().items() if tablename is not None}
		dropped = 0
		batchvalues = []
		for rowindex, row in self.table.get(dirname, processes, self.rows_to_import):
			if rowindex == 0:
				column_names = []
				for colname, colvalue in row.items():
					column_names.append(colname)

			key = self.table.extract_key(row)
			if key in self.imported_keys:
				print(f"\rWARNING : Duplicate primary key {key} in table {self.table.name}. The row has been dropped")
				dropped += 1
				continue

			check_fail = False
			for colname, process in foreign_checks.items():
				if (row[colname], ) not in process.imported_keys:
					#print(f"\rWARNING : Foreign key {row[colname]} in column {colname} of table {self.table.name} has no target in table {process.table.name}. The row has been dropped")
					check_fail = True
			if check_fail:
				dropped += 1
				continue


			column_values = []
			for colname, colvalue in row.items():
				column_values.append(convert_insert(colvalue, columns[colname][0]))
			batchvalues.append("(" + ",".join(column_values) + ")")

			if len(batchvalues) >= BATCH_SIZE:
				# Let’s make the hopeful guess there is nothing dangerous in our data for now
				valuestring = ',\n'.join(batchvalues)
				query = f"INSERT INTO {self.table.name} ({','.join(column_names)}) VALUES {valuestring};";
				connection.execute(query)
				connection.commit()

				imported += len(batchvalues)
				print(f"\r{imported} rows inserted", end="")
				batchvalues.clear()

			self.imported_keys.add(key)
		
		if len(batchvalues) > 0:
			valuestring = ',\n'.join(batchvalues)
			query = f"INSERT INTO {self.table.name} ({','.join(column_names)}) VALUES {valuestring};";
			connection.execute(query)
			connection.commit()
			imported += len(batchvalues)
			batchvalues.clear()

		print(f"\r{imported} rows inserted")
		print(f"{dropped} rows dropped")

	def selected(self, rowindex):
		"""Check whether the given rowindex is going to be imported"""
		return self.rows_to_import is None or key in self.rows_to_import


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





########## Main script

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_directory", help="Directory where the dataset files are contained")
	parser.add_argument("--host", help="MySQL server address", default="127.0.0.1")
	parser.add_argument("--port", type=int, help="MySQL server port", default=3306)
	parser.add_argument("-u", "--user", help="MySQL database user name", default="")
	parser.add_argument("-p", "--password", help="MySQL database user password", default="")
	parser.add_argument("-d", "--database", help="Target MySQL database name")
	parser.add_argument("-t", "--truncate", type=int, help="Only keep this number of rows in each table", default=0)
	args = parser.parse_args()

	input_directory = args.dataset_directory
	connection = MySQLConnectionWrapper(args.host, args.port, args.database, args.user, args.password)

	if args.truncate > 0:
		for table in table_classes.values():
			table.truncate(args.truncate)

	processes = {}
	for tablename, table in table_classes.items():
		processes[tablename] = TableImport(connection, table)

	print("\n------ Resolving import order")
	import_order = resolve_import_order(processes)
	print(f"Import order : {', '.join(import_order)}")

	print("\n------ Dropping existing tables")
	for tablename in reversed(import_order):
		processes[tablename].drop()

	print("\n------ Importing data")
	for tablename in import_order:
		print("--- Importing table", tablename)
		processes[tablename].run(processes, input_directory)

	connection.close()