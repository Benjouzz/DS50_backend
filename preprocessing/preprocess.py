"""
Main utility to run the preprocessing pipeline
$ preprocess --help for usage information

@author Grégori MIGNEROT
"""


import os
import time
import json
import pickle
import argparse
import datetime
import concurrent.futures

import pipeline.filter as filterstep
import pipeline.review as reviewstep
import pipeline.tag as tagstep
import pipeline.user as userstep
import pipeline.final as finalstep
import pipeline.database as importstep

from util.tag import Hierarchy
from util.data import read_csv_rows, read_json_rows
from util.pipeline import PreprocessingPipelineConfig, PreprocessingLogger


# Default config file, can be set with option -c <configfile>
CONFIG_FILE = "preprocessing_config.json"

# General pipeline steps
STEPS = ["filter", "review", "tag", "user", "final", "import"]

# Pipeline unitary steps
SUBSTEPS = {
#   "step.substep"         function to call
	"filter.books":        filterstep.filter_books,
	"filter.interactions": filterstep.filter_interactions,
	"filter.reviews":      filterstep.filter_reviews,

	"review.reviews":      reviewstep.fix_reviews,
	"tag.categorize":      tagstep.categorize,
	"user.generate":       userstep.generate_users,

	# "final.reviews":     finalstep.process_reviews,
	"final.interactions":  finalstep.process_interactions,
	"final.books":         finalstep.process_books,
	"final.authors":       finalstep.process_authors,
	"final.wrote":         finalstep.process_wrote,
	"final.contains":      finalstep.process_contains,

	"import.order":        importstep.resolve_import_order,
	"import.drop":         importstep.drop_tables,
	"import.book":         importstep.import_table,
	"import.tag":          importstep.import_table,
	"import.tagged":       importstep.import_table,
	"import.author":       importstep.import_table,
	"import.wrote":        importstep.import_table,
	"import.series":       importstep.import_table,
	"import.contains":     importstep.import_table,
	"import.interaction":  importstep.import_table,
	# "import.review":     importstep.import_table,
	"import.user":         importstep.import_table,
}

# Identifiers of the arguments and return values of each substep function
SUBSTEPS_IO = {
#    substep identifier    ([arg1_id, arg2_id, ...], [return1_id, return2_id, ...])
	"filter.books":        (["pipeline.books", "files.filter.input.books", "files.filter.output.books", "files.filter.input.series", "files.filter.output.series", "files.filter.input.authors", "files.filter.output.authors"],
							["filter.books.book_ids"]),
	"filter.interactions": (["pipeline.users", "filter.books.book_ids", "files.filter.input.interactions", "files.filter.output.interactions", "files.filter.input.bookmap", "files.filter.output.bookmap", "files.filter.input.usermap", "files.filter.output.usermap"],
							["filter.interactions.user_sample"]),
	"filter.reviews":      (["filter.interactions.user_sample", "usermap.jsontocsv", "filter.books.book_ids", "files.filter.input.reviews", "files.filter.output.reviews"],
							[]),

	"review.reviews":      (["files.review.input.reviews", "files.review.output.reviews", "pipeline.review_training"],
							[]),
	"tag.categorize":      (["files.tag.input.books", "files.tag.input.authors", "files.tag.input.series", "files.tag.output.tags", "files.tag.output.booktags", "files.tag.output.association", "files.tag.output.unassociated", "tag.hierarchy"],
							[]),
	"user.generate":       (["usermap.csvtojson", "bookmap.csvtojson", "files.user.input.interactions", "files.user.input.booktags", "files.user.output.users", "tag.hierarchy"],
							[]),

	# "final.reviews":     (["usermap.jsontocsv", "files.final.input.reviews", "files.final.output.reviews"],
	#						["final.book_reviews", "final.review_ratings"]),
	"final.interactions":  (["usermap.jsontocsv", "bookmap.csvtojson", "files.final.input.reviews", "files.final.input.interactions", "files.final.output.interactions"],
							["final.book_ratings", "final.book_reviews"]),
	"final.books":         (["final.book_ratings", "final.book_reviews", "files.final.input.booktags", "files.final.input.books", "files.final.output.books"],
							["final.author_books", "final.author_ratings", "final.author_reviews", "final.series_books"]),
	"final.authors":       (["final.author_ratings", "final.author_reviews", "files.final.input.authors", "files.final.output.authors"],
							[]),
	"final.wrote":         (["final.author_books", "files.final.output.wrote"],
							[]),
	"final.contains":      (["final.series_books", "files.final.output.contains"],
							[]),

	"import.order":        None,  # Special substep because we need dynamic dependencies for table imports
	"import.drop":         (["database.processes"], []),
	"import.book":         (["database.processes", "tablename.book", "files.import.input.books"],
							["database.processes/" + importstep.TableName.Book]),
	"import.tag":          (["database.processes", "tablename.tag", "files.import.input.tags"],
							["database.processes/" + importstep.TableName.Tag]),
	"import.tagged":       (["database.processes", "tablename.tagged", "files.import.input.booktags"],
							["database.processes/" + importstep.TableName.Tagged]),
	"import.author":       (["database.processes", "tablename.author", "files.import.input.authors"],
							["database.processes/" + importstep.TableName.Author]),
	"import.wrote":        (["database.processes", "tablename.wrote", "files.import.input.wrote"],
							["database.processes/" + importstep.TableName.Wrote]),
	"import.series":       (["database.processes", "tablename.series", "files.import.input.series"],
							["database.processes/" + importstep.TableName.Series]),
	"import.contains":     (["database.processes", "tablename.contains", "files.import.input.contains"],
							["database.processes/" + importstep.TableName.Contains]),
	"import.interaction":  (["database.processes", "tablename.interaction", "files.import.input.interactions"], 
							["database.processes/" + importstep.TableName.Interaction]),
	# "import.review":     (["database.processes", "tablename.review", "files.import.input.reviews"],
	# 						["database.processes/" + importstep.TableName.Review]),
	"import.user":         (["database.processes", "tablename.user", "files.import.input.users"],
							["database.processes/" + importstep.TableName.User]),
}

# Substep dependency graph
DEPENDENCIES = {
	"filter.books":        set(),
	"filter.interactions": {"filter.books"},
	"filter.reviews":      {"filter.interactions"},

	"review.reviews":      {"filter.reviews"},
	"tag.categorize":      {"filter.books"},
	"user.generate":       {"filter.interactions", "tag.categorize"},

	# "final.reviews":     {"review.reviews"},
	"final.interactions":  {"review.reviews", "filter.interactions"},
	"final.books":         {"final.interactions", "tag.categorize"},
	"final.authors":       {"final.books"},
	"final.wrote":         {"final.books"},
	"final.contains":      {"final.books"},

	"import.order":        set(),
	"import.drop":         {"__WAIT"},  # Do not start right away like without any dependency, rather only when needed
	"import.book":         {"import.drop", "final.books"},
	"import.tag":          {"import.drop", "tag.categorize"},
	"import.tagged":       {"import.drop", "tag.categorize"},
	"import.author":       {"import.drop", "final.authors"},
	"import.wrote":        {"import.drop", "final.wrote"},
	"import.series":       {"import.drop", "filter.books"},
	"import.contains":     {"import.drop", "final.contains"},
	"import.interaction":  {"import.drop", "final.interactions"},
	# "import.review":     {"import.drop", "final.reviews"},
	"import.user":         {"import.drop", "user.generate"},
}

# Substep associated to each table
TABLE_STEPS = {
	importstep.TableName.Book:        "import.book",
	importstep.TableName.Tag:         "import.tag",
	importstep.TableName.Tagged:      "import.tagged",
	importstep.TableName.Author:      "import.author",
	importstep.TableName.Wrote:       "import.wrote",
	importstep.TableName.Series:      "import.series",
	importstep.TableName.Contains:    "import.contains",
	importstep.TableName.Interaction: "import.interaction",
	# importstep.TableName.Review:    "import.review",
	importstep.TableName.User:        "import.user",
}


def run_step(config, log, step:str):
	"""Run a single pipeline step"""
	log.subtitle(f"Running only step {step}")
	substeps = [substep for substep in SUBSTEPS.keys() if substep.startswith(step + ".")]
	run_steps(config, log, substeps)

def run_to(config, log, last_step:str):
	"""Run the pipeline until the given step (included)"""
	log.subtitle(f"Running until step {last_step}")
	substeps = []
	for step in STEPS:
		substeps.extend([substep for substep in SUBSTEPS.keys() if substep.startswith(step + ".")])
		if step == last_step:
			break
	run_steps(config, log, substeps)

def load_state(config, log):
	"""Restart the pipeline from a saved state"""
	with open(config.state_file, "rb") as save:
		savedata = pickle.load(save)

	run_steps(config, log, savedata["substeps"], savedata["common_information"], savedata["pending_substeps"], savedata["finished_substeps"])


def save_state(log, substeps, finished_substeps, common_information):
	"""Save the pipeline state to be able to eventually start it from where it crashed"""
	pending_substeps = set(substeps) - set(finished_substeps) - {"import.order"}
	log.title("Quitting on exception, saving state")
	log.print(f"Pending substeps  : {list(pending_substeps)}")
	log.print(f"Finished substeps : {list(finished_substeps)}")

	savedata = {
		"substeps": substeps,
		"pending_substeps": pending_substeps,
		"finished_substeps": finished_substeps,
		"common_information": common_information,
	}

	with open(config.state_file, "wb") as save:
		pickle.dump(savedata, save)


def build_common_information(config):
	"""Initialise the common substeps I/O data"""
	info = {}

	# `pipeline` namespace : profile configuration
	profile = config.get_profile()
	info["pipeline.books"] = profile["books"]
	info["pipeline.users"] = profile["users"]
	info["pipeline.directory"] = profile["directory"]
	info["pipeline.review_training"] = profile["review_training"]

	# `files` namespace : file paths for each step, input and output
	for step, ioparts in config.files.items():
		for iopart, categories in ioparts.items():
			for category, filename in categories.items():
				info[f"files.{step}.{iopart}.{category}"] = filename

	with open("category-hierarchy.json", "r", encoding="utf-8") as jsonfile:
		hierarchy = Hierarchy.load(json.load(jsonfile))
	info["tag.hierarchy"] = hierarchy

	# `database` namespace : database server and processes information
	connection = importstep.MySQLConnectionWrapper(config.server["host"], config.server["port"], config.server["database"], config.server["username"], config.server["password"])
	processes = {}
	for tablename, table in importstep.table_classes.items():
		processes[tablename] = importstep.TableImport(connection.astuple(), table)
	info["database.processes"] = processes
	info["database.connection"] = connection.astuple()

	# `import.tablename` namespace : table names with identifiers
	for tablename, substep in TABLE_STEPS.items():
		info[substep.replace("import.", "tablename.")] = tablename

	return info

def load_usermap(config):
	"""Load the user ID mappings (csv <-> json)
	   Return json<str> -> csv<int> and csv<int> -> json<str>, respectively"""
	filename = config.files[STEPS[0]]["output"]["usermap"]
	user_jsontocsv = {}
	user_csvtojson = {}
	for rowindex, row in read_csv_rows(filename):
		user_jsontocsv[row["user_id"]] = int(row["user_id_csv"])
		user_csvtojson[int(row["user_id_csv"])] = row["user_id"]
	return user_jsontocsv, user_csvtojson

def load_bookmap(config):
	"""Load the book ID mappings (csv <-> json)
	   Return json<int> -> csv<int> and csv<int> -> json<int>, respectively"""
	filename = config.files[STEPS[0]]["output"]["bookmap"]
	book_jsontocsv = {}
	book_csvtojson = {}
	for rowindex, row in read_csv_rows(filename):
		book_jsontocsv[int(row["book_id"])] = int(row["book_id_csv"])
		book_csvtojson[int(row["book_id_csv"])] = int(row["book_id"])
	return book_jsontocsv, book_csvtojson

def resolve_arguments(argnames, common_information):
	"""Retrieve the arguments values from the substep I/O data with the list of arguments names"""
	arglist = []
	for name in argnames:
		if name in common_information:
			arglist.append(common_information[name])
		elif name.startswith("usermap."):
			common_information["usermap.jsontocsv"], common_information["usermap.csvtojson"] = load_usermap(config)
			arglist.append(common_information[name])
		elif name.startswith("bookmap."):
			common_information["bookmap.jsontocsv"], common_information["bookmap.csvtojson"] = load_bookmap(config)
			arglist.append(common_information[name])
		else:
			raise KeyError(f"Argument key {name} not recognized")
	return arglist

def set_result(step, result, common_information):
	"""Set the return values of the given step in the common substep I/O data"""
	if result is None:
		result = []
	elif not isinstance(result, (list, tuple)):
		result = [result]

	for id, value in zip(SUBSTEPS_IO[step][1], result):
		if "/" in id:
			variable, key = id.split("/")
			common_information[variable][key] = value
		else:
			common_information[id] = value


def run_steps(config, log, substeps, common_information=None, pending_substeps=None, finished_substeps=None):
	"""Run the given pipeline substeps"""
	substeps = set(substeps)
	log.print(f"Substeps to run : {', '.join(substeps)}")

	# Establish the dynamic dependencies between the import tasks
	if "import.order" in substeps:
		dependencies = importstep.table_dependencies()
		for tablename, dependencies in dependencies.items():
			DEPENDENCIES[TABLE_STEPS[tablename]].update({TABLE_STEPS[dep] for dep in dependencies})

	# Initialize the global information
	common_information = build_common_information(config) if common_information is None else common_information
	pending_substeps = substeps - {"import.order"} if pending_substeps is None else set(pending_substeps)
	current_substeps = {}  # {substep_id: future, ...}
	finished_substeps = set() if finished_substeps is None else set(finished_substeps)

	# From there on, each substep is run as a separate process in a ProcessPoolExecutor
	# This allows the actual parallelization of a lot of substeps, resulting in a ≈2× acceleration of the whole process
	# As the future objects lack a bit in control and sequencing features, we manage the scheduling ourselves
	# At each check, finished substeps are pulled out and newly available substeps (with all dependencies met) are pushed in
	# We do this at a small delay, no need to crush the CPU to schedule tasks that take at least several minutes each anyway
	starttime = time.time()
	with concurrent.futures.ProcessPoolExecutor() as executor:
		try:
			empty_rounds = 0
			while len(pending_substeps) > 0 or len(current_substeps) > 0:
				# Pull out finished substeps
				for step, future in current_substeps.items():
					if future.done():  # Substep finished
						log.status(f"SUBSTEP {step} FINISHED")
						try:
							result = future.result()
							set_result(step, result, common_information)
							finished_substeps.add(step)
						except Exception as exc:
							# In case of exception in the substep, the exception is raised by future.result()
							# Then save the pipeline state
							log.exception(exc)
							executor.shutdown(wait=False, cancel_futures=False)
							save_state(log, substeps, finished_substeps, common_information)
							return

				# Push in available substeps
				for step in pending_substeps:
					# Dependencies that are in `substeps` (to be executed here, in case on single step run)
					# Including __WAIT that signals to wait until the substep is needed
					# And excluding finished substeps, so in the end,
					# remaining_dependencies empty -> the substep is available
					remaining_dependencies = tuple((DEPENDENCIES[step] & (substeps | {"__WAIT"})) - finished_substeps)
					if len(remaining_dependencies) == 0:
						log.status(f"STARTING SUBSTEP {step}")
						arguments = resolve_arguments(SUBSTEPS_IO[step][0], common_information)

						# We give each task a sub-logger that will only report when closed, so the console output is scrambled but not the log file
						current_substeps[step] = executor.submit(SUBSTEPS[step], log.branch(), *arguments)
					else:
						# Get the remaining dependencies’ dependencies, and check whether each of them has only __WAIT left as dependency
						# In that case, those dependencies are needed now so we can stop waiting and run them
						dependencies_dependencies = {dep: (DEPENDENCIES[dep] & (substeps | {"__WAIT"})) - finished_substeps for dep in remaining_dependencies if dep != "__WAIT"}
						if all([len(deps) == 1 and tuple(deps)[0] == "__WAIT" for deps in dependencies_dependencies.values()]):
							for dep in dependencies_dependencies:
								DEPENDENCIES[dep].remove("__WAIT")

				# Updating the pending and current substeps according to the changes
				pending_substeps -= set(current_substeps.keys())
				current_substeps = {step: future for step, future in current_substeps.items() if step not in finished_substeps}
				
				# Theoretically, there should be no more than a single empty round in a row
				# If there is (here 5 because why not), it will never go forward again
				# This is probably a dependency cycle, so stop here and save the pipeline state
				if len(current_substeps) == 0:
					empty_rounds += 1
				if empty_rounds > 5:
					log.error("More than 5 empty rounds in a row : possible dependency loop.")
					log.print(f"All substeps to run : {', '.join(substeps)}")
					log.print(f"Pending substeps  : {', '.join(pending_substeps)}")
					log.print(f"Current substeps  : {', '.join(current_substeps.keys())}")
					log.print(f"Finished substeps : {', '.join(finished_substeps)}")
					save_state(log, substeps, finished_substeps, common_information)
					return

				time.sleep(1)  # Delay between checks
		except Exception as exc:
			# Save the pipeline state in case of exception
			log.exception(exc)
			executor.shutdown(wait=False, cancel_futures=False)
			save_state(log, substeps, finished_substeps, common_information)
			return

	endtime = time.time()
	log.title("Pipeline report")
	log.section(f"Pipeline finished in {endtime - starttime :.3f} seconds")
	log.section("Substeps accomplished :")
	log.print(", ".join(finished_substeps))



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run our DS50 project preprocessing pipeline")
	parser.add_argument("profile", help="Pipeline profile to use, as defined in the configuration file")
	parser.add_argument("-c", "--configfile", help="Configuration file to use", default=CONFIG_FILE)
	parser.add_argument("-s", "--step", help=f"Run a single step (available steps : {', '.join(STEPS)})")
	parser.add_argument("-t", "--to", help=f"Run the pipeline until the given step (available steps : {', '.join(STEPS)})")
	parser.add_argument("-d", "--dirty", action="store_true", help="Do not clean the output directory before running from the start (using -s won’t clean it by default)")
	parser.add_argument("-l", "--loadstate", action="store_true", help="Start from the last failure")
	parser.add_argument("--diagram", action="store_true", help="Generate the database diagram for https://dbdiagram.io/d")
	args = parser.parse_args()

	# Initialize the configuration and logger
	config = PreprocessingPipelineConfig(args.configfile, STEPS)

	log_filename = os.path.join(config.log_directory, datetime.datetime.now().strftime(args.profile + "-%Y%m%d-%H%M%S.log"))
	log = PreprocessingLogger(log_filename).start()

	if args.diagram:
		importstep.generate_diagram(log)
		exit()

	log.title("Preparing the pipeline")
	log.print(f"Logging in file {log_filename}")

	log.section("Preparing the output directory")
	config.set_profile(args.profile)
	if not os.path.exists(config.output_dir):
		os.mkdir(config.output_dir)
		log.print(f"Created directory {config.output_dir}")

	if args.step is None and not args.dirty and not args.loadstate:
		for filename in os.listdir(config.output_dir):
			os.remove(os.path.join(config.output_dir, filename))
		log.print(f"Emptied directory {config.output_dir}")


	if args.step is not None:
		run_step(config, log, args.step)
	elif args.to is not None:
		run_to(config, log, args.to)
	elif args.loadstate:
		load_state(config, log)
	else:
		run_to(config, log, STEPS[-1])

	log.close()
