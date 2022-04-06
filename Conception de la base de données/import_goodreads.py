import os
import sys
import csv
import json
import sqlite3
import pprint
import random
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




class Table (object):
	"""Superclass for DB tables.
	   Needs to be subclassed once for each table"""
	def __init__(self, name):
		self.name = name
		self.filters = []
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


class Filter (object):
	"""Base class for row filters"""
	def resolve(self, table, processes, dirname):
		"""Apply the filter and return the filtered row indices
		   - int[] table    : table object to apply the filter onto
		   - dict processes : all table import process objects
		   - str dirname    : dataset root directory"""
		NotImplemented


class RandomFilter (Filter):
	"""Selects some amount of rows at random"""
	def __init__(self, num_select):
		self.num_select = num_select

	def resolve(self, table, processes, dirname):
		selfprocess = processes[table.name]

		# Optimisation : we need to read the entire file first to count the rows and get
		# the row indices to sample from. If some row indices have already been selected,
		# we can just sample from them
		if selfprocess.rows_to_import is not None:
			return set(random.sample(tuple(selfprocess.rows_to_import), self.num_select))
		else:
			available_rows = []
			for rowindex, row in table.get(dirname, processes, selfprocess.rows_to_import):
				available_rows.append(rowindex)
			return set(random.sample(available_rows, self.num_select))

	def __repr__(self):
		return f"{self.__class__.__name__}({repr(self.num_select)})"


class MatchingFilter (Filter):
	"""Selects only rows that match any value in a foreign column.
	   - str self_column    : column to filter on
	   - str foreign_table  : name of the foreign_table to match
	   - str foreign_column : only the rows that match any value in this column of the
	                          foreign table will be kept"""
	def __init__(self, self_column, foreign_table, foreign_column):
		self.self_column = self_column
		self.foreign_table = foreign_table
		self.foreign_column = foreign_column

	def resolve(self, table, processes, dirname):
		self_process = processes[table.name]
		foreign_process = processes[self.foreign_table]
		# Need to resolve the foreign table filters first to avoid keeping useless rows
		foreign_process.resolve_filters(dirname, processes)

		# Get all relevant values in the foreign column
		foreign_selected = set()
		for rowindex, row in foreign_process.table.get(dirname, processes, foreign_process.rows_to_import):
			#print(f"SELECT {type(row[self.foreign_column])} : {row[self.foreign_column]}")
			foreign_selected.add(row[self.foreign_column])

		# Only keep the row indices where the local column matches any value of the foreign column
		self_selected = set()
		for rowindex, row in table.get(dirname, processes, self_process.rows_to_import):
			#print(f"CHECK {type(row[self.self_column])} :", rowindex, row[self.self_column], row[self.self_column] in foreign_selected)
			if row[self.self_column] in foreign_selected:
				self_selected.add(rowindex)

		return self_selected

	def __repr__(self):
		return f"{self.__class__.__name__}({repr(self.self_column)}, {repr(self.foreign_table)}, {repr(self.foreign_column)})"



# DB type aliases
Int = "INTEGER"
String = "TEXT"
Real = "REAL"
Bool = "BOOLEAN"



def convert_value(value, type):
	"""Convert a value from the base file to a python representation"""
	if type == Int:
		return int(value) if value != "" else None
	elif type == Real:
		return float(value) if value != "" else None
	elif type == Bool:
		return bool(value) if value != "" else None
	elif type == String:
		return value

def convert_insert(value, type):
	"""Convert a value to its SQL representation"""
	if value is None:
		return "NULL"
	elif type in (Int, Real):
		return str(value)
	elif type == Bool:
		return "TRUE" if value else "FALSE"
	elif type == String:
		return "'" + value.replace("'", "''") + "'"

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

# Table names enumeration
class TableName:
	Book = "BOOK"
	Work = "WORK"
	Author = "AUTHOR"

	Wrote = "WROTE"


########## Table definitions

class BookTable (Table):
	def columns(self):
		return {
			#name                 type    pk     fk
			"book_id":           (Int,    True,  None),
			"country_code":      (String, False, None),
			"description":       (String, False, None),
			"format":            (String, False, None),
			"image_url":         (String, False, None),
			"is_ebook":          (Bool,   False, None),
			"language_code":     (String, False, None),
			"num_pages":         (Int,    False, None),
			"publication_year":  (Int,    False, None),
			"publication_month": (Int,    False, None),
			"publisher":         (String, False, None),
			"title":             (String, False, None),
			"work_id":           (String, False, TableName.Work)}

	def get(self, dirname, processes, selectedrows=None):
		for rowindex, row in read_json_rows(os.path.join(dirname, "goodreads_books.json"), selectedrows, self.truncation):
			importrow = {}
			for colname, (type, pk, fk) in self.columns().items():
				importrow[colname] = convert_value(row[colname], type)
			yield (rowindex, importrow)


class AuthorTable (Table):
	def columns(self):
		return {
			"author_id": (Int,    True,  None),
			"name":      (String, False, None)}

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


# Map the table names to their related table object
table_classes = {
	TableName.Book: BookTable(TableName.Book),
	TableName.Wrote: WroteTable(TableName.Wrote),
	TableName.Author: AuthorTable(TableName.Author),
}


# Insertions to commit at once
BATCH_SIZE = 200




class TableImport (object):
	"""Holds the state of a table import"""

	def __init__(self, connection, table):
		self.connection = connection
		self.table = table
		self.resolved_filters = False
		self.rows_to_import = None
		self.imported_keys = set()

		if self.table.truncation is not None:
			self.rows_to_import = set(range(self.table.truncation))

	def resolve_filters(self, processes, dirname):
		"""Resolve the filters on the associated table before importing"""
		if not self.resolved_filters:  # Only apply them once
			for filter in self.table.filters:
				print(f"Resolving filter {repr(filter)} on process {self.table.name}")
				selected_keys = filter.resolve(self.table, processes, dirname)
				if self.rows_to_import is None:
					self.rows_to_import = set(selected_keys)
				else:  # Intersection with the superset
					self.rows_to_import &= selected_keys
			self.resolved_filters = True

	def run(self, processes, dirname):
		"""Run the import process (table creation and data insertion)"""
		self.create()
		self.insert(processes, dirname)

	def create(self):
		"""Drop the table if it already exists, and create it"""
		columns = ""
		constraints = ""

		constraints += f"PRIMARY KEY ({','.join(self.table.primary_key())}),\n"

		for colname, (type, primary_key, foreign_key) in self.table.columns().items():			
			columns += f"{colname} {type},\n"

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
						constraints += f"FOREIGN KEY ({colname}) REFERENCES {foreign_key}({foreign_pk}),\n"

		if constraints != "":
			constraints = constraints.rstrip("\n,")
		else:
			columns = columns.rstrip("\n,")

		# Let’s make the hopeful guess there is nothing dangerous in our data for now
		connection.execute(f"DROP TABLE IF EXISTS {self.table.name};")
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

		for rowindex, row in self.table.get(dirname, processes, self.rows_to_import):
			key = self.table.extract_key(row)
			if key in self.imported_keys:
				print(f"\rWARNING : Duplicate primary key {key} in table {self.table.name}. The row has been dropped")
				continue

			column_names = []
			column_values = []
			for colname, colvalue in row.items():
				column_names.append(colname)
				column_values.append(convert_insert(colvalue, columns[colname][0]))

			query = f"INSERT INTO {self.table.name} ({','.join(column_names)}) VALUES ({','.join(column_values)});";
			connection.execute(query)

			imported += 1
			self.imported_keys.add(key)
			if imported % BATCH_SIZE == 0:
				connection.commit()
				print(f"\r{imported} rows inserted", end="")
		connection.commit()
		print(f"\r{imported} rows inserted")

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


######### Row filters definition

table_classes[TableName.Book].truncate(1000)
table_classes[TableName.Book].add_filters(
	RandomFilter(20))

table_classes[TableName.Wrote].add_filters(
	MatchingFilter("book_id", TableName.Book, "book_id"))

table_classes[TableName.Author].add_filters(
	MatchingFilter("author_id", TableName.Wrote, "author_id"))


########## Main script

if __name__ == "__main__":
	input_directory = sys.argv[1]
	connection = SQLiteConnectionWrapper("goodreads.db")

	processes = {}
	for tablename, table in table_classes.items():
		processes[tablename] = TableImport(connection, table)

	print("\n------ Resolving filters")
	for tablename, process in processes.items():
		process.resolve_filters(processes, input_directory)

	print("\n------ Resolving import order")
	import_order = resolve_import_order(processes)
	print(f"Import order : {', '.join(import_order)}")

	print("\n------ Importing data")
	for tablename in import_order:
		print("--- Importing table", tablename)
		processes[tablename].run(processes, input_directory)

	connection.close()