import io
import os
import sys
import json
import traceback


class PreprocessingPipelineConfig (object):
	def __init__(self, filename, steps):
		with open(filename, "r", encoding="utf-8") as configfile:
			dic = json.load(configfile)

		self.configs = dic["configurations"]
		self.steps = steps

		self.dataset_path = dic["files"]["dataset"]
		self.log_directory = dic["files"]["logs"]
		self.state_file = dic["files"]["statefile"]

		self.filedata = dic["files"]

	def set_profile(self, name):
		self.configname = name
		self.output_dir = self.configs[name]["directory"]
		self.server = self.configs[name]["server"]

		self.files = {}

		input_files = {category: os.path.join(self.dataset_path, filename) for category, filename in self.filedata["input"].items()}
		output_files = {category: os.path.join(self.output_dir, filename) for category, filename in self.filedata["input"].items()}
		self.files[self.steps[0]] = {"input": input_files, "output": output_files}

		for step in self.steps[1:]:
			input_files = output_files.copy()
			output_files = output_files.copy()
			if step in self.filedata:
				for identifier, filename in self.filedata[step].items():
					output_files[identifier] = os.path.join(self.output_dir, filename)
			self.files[step] = {"input": input_files, "output": output_files}

	def get_profile(self):
		return self.configs[self.configname]


class PreprocessingLogger (object):
	def __init__(self, filename):
		self.filename = filename
		self.errors = 0

	def start(self):
		self.content = io.StringIO()
		return self

	def flush(self):
		self.output = self.content.getvalue()
		with open(self.filename, "a", encoding="utf-8") as logfile:
			logfile.write(self.output)
		self.content.close()
		self.content = io.StringIO()

	def close(self):
		self.output = self.content.getvalue()
		with open(self.filename, "a", encoding="utf-8") as logfile:
			logfile.write(self.output)
		self.content.close()
		self.content = None

	def branch(self):
		self.flush()
		return PreprocessingLogger(self.filename).start()

	def write(self, text):
		self.content.write(text)
		sys.stdout.write(text)

	def print(self, text):
		self.write(text + "\n")

	def status(self, text):
		sys.stdout.write(text + "\n")

	def title(self, text):
		self.print(f"\n\n======== {text.upper()}")

	def subtitle(self, text):
		self.print(f"\n==== {text}")

	def section(self, text):
		self.print(f"== {text}")

	def warning(self, text):
		self.print(f"WARNING : {text}")

	def error(self, text):
		self.errors += 1
		self.print(f"ERROR : {text}")

	def exception(self, exc):
		self.errors += 1
		traceback.print_exc(file=self)
