import os
import sys
import json
import pprint
from nltk.tag import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from util.tag import *



if __name__ == "__main__":
	startfrom = -1
	hierarchy_titles = set()
	if len(sys.argv) > 1:
		if sys.argv[1] == "-l":
			with open("category-hierarchy.json", "r", encoding="utf-8") as f:
				hierarchy = Hierarchy.load(json.load(f))
				startfrom = hierarchy.maxid()
		elif sys.argv[1] == "-f":
			with open("category-hierarchy.json", "r", encoding="utf-8") as f:
				hierarchy = Hierarchy.load(json.load(f))
			base_titles = set()
			with open("category-hierarchy-base.txt", "r", encoding="utf-8") as f:
				for line in f:
					if line.strip() != "":
						base_titles.add(line.split(",")[0].strip().lower())
			for id, entry in tuple(hierarchy.entries.items()):
				if entry.title not in base_titles:
					hierarchy.remove(id)
				else:
					hierarchy_titles.add(entry.title)

	else:
		hierarchy = Hierarchy()

	current_id = 0
	super_stack = []
	with open("category-hierarchy-base.txt", "r", encoding="utf-8") as infile:
		try:
			for line in infile:
				if line.strip() == "":
					continue
				elif line.strip() == "[STOP]":
					break

				indent = len(line) - len(line.lstrip())
				if indent > len(super_stack):
					super_stack.append(current_id - 1)
				while indent < len(super_stack):
					super_stack.pop()

				title = line.split(",")[0].strip().lower()

				if current_id > startfrom and title not in hierarchy_titles:
					favorite_select = "!" in line
					content = line.replace("!", "").strip()
					textnames = [name.strip() for name in content.split(",")]
					names = []
					for textname in textnames:
						basetokens = Name.parse(textname)
						finaltokensets = [set()]
						for word, pos, hint in basetokens:
							chosen_synsets = None
							print("\n\n")
							print(f"In text name {textname}, token {(word, pos, hint)}")
							if word in ENGLISH_WORDS:
								synsets = wordnet.synsets(word)
								print("With identified part-of-speech : ")
								for i, synset in enumerate(synsets):
									if synset.pos() == pos:
										print(f"\t{i+1} - {synset.name()} : {synset.definition()}")
								print("With different part-of-speech : ")
								for i, synset in enumerate(synsets):
									if synset.pos() != pos:
										print(f"\t{i+1} - {synset.name()} : {synset.definition()}")
								print("Enter indices separated by spaces, nothing to skip, or a POS tag to just change the POS without selecting a synset")

								response = input("> ").strip().lower()
								if response.replace(" ", "").isdigit():
									chosen_synsets = [synsets[int(index.strip())-1] for index in response.split() if int(index.strip())-1 < len(synsets)]
								elif response in (wordnet.NOUN, wordnet.ADJ, wordnet.ADV, wordnet.VERB, wordnet.ADJ_SAT):
									pos = response.lower()
							else:
								print("Word not found in WordNet")

							if chosen_synsets is not None:
								newsets = []
								for synset in chosen_synsets:
									for tokenset in finaltokensets:
										newset = tokenset.copy()
										newset.add((word, synset.pos(), synset.name()))
										newsets.append(newset)
								finaltokensets = newsets
							else:
								for tokenset in finaltokensets:
									tokenset.add((word, pos, hint))

						for tokenset in finaltokensets:
							names.append(Name(tokenset))

					entry = HierarchyItem(current_id, textnames[0], names, super_stack[-1] if len(super_stack) > 0 else None, [], len(super_stack), tuple(super_stack) + (current_id, ), favorite_select)
					hierarchy.add(current_id, entry)

					if len(super_stack) > 0:
						hierarchy.get(super_stack[-1]).subclasses.append(current_id)

				elif title in hierarchy_titles:
					entry = hierarchy.get_title(title)
					hierarchy.move(entry.id, current_id)

				current_id += 1
		except KeyboardInterrupt:
			pass

	with open("category-hierarchy.json", "w", encoding="utf-8") as outfile:
		json.dump(hierarchy.save(), outfile, indent=4)