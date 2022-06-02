import csv
import json

def read_json_rows(filename):
	with open(filename, "r", encoding="utf-8") as f:
		for i, row in enumerate(f):
			yield (i, json.loads(row))

def read_csv_rows(filename):
	with open(filename, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for i, row in enumerate(reader):
			yield (i, row)