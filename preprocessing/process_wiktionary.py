"""
DS50 project, preprocessing pipeline

Parse a Wiktionary XML dump into a usable thesaurus representation,
in order to determine relationships between words with different
part of speech in the book tag categorization

@author Grégori MIGNEROT
"""

import re
import json
import pprint
import xml.sax
import mwparserfromhell as mw
from dataclasses import dataclass
from xml.sax.handler import ContentHandler
from collections.abc import Iterable
from functools import singledispatchmethod


############## INITIALIZE RELATIONSHIPS

RELATIONSHIP_CODES = {}

@dataclass
class _Relationship:
	code:str           # Compact code to represent the relationship
	samepos:bool       # Whether the relationship only apply to words with the same part of speech
	wordhierarchy:int  # Lexical hierarchical relationship (-1 = target more generic than self, 0 = horizontal, +1 = target more specific than self, None = ambiguous, ambivalent or no information)
	sethierarchy:int   # Sense hierarchical relationship   (-1 = target container of self,      0 = horizontal, +1 = target part of self,            None = ambiguous, ambivalent or no information)
	objhierarchy:int   # Object hierarchical relationship  (-1 = target category of self,       0 = horizontal, +1 = target instance of self,        None = ambiguous, ambivalent or no information)
	equivalence:bool   # Whether the relationship makes a full equivalence between both words
	symetric:object = None  # Symetric relationship

	def __post_init__(self):
		RELATIONSHIP_CODES[self.code] = self

	def __eq__(self, rel):
		return self.code == rel.code or self.code == rel

	def __ne__(self, rel):
		return self.code != rel.code and self.code != rel

	def __hash__(self):
		return hash(self.code)

	def __call__(self, entry):
		if self.code in entry.relationships:
			return entry.relationships[self]
		else:
			return set()

# Define the actual relationships
Compound =     _Relationship("cpd", False, None, None, None, False)
Component =    _Relationship("cpn", False, None, None, None, False)
Derived =      _Relationship("der", False, None, None, None, False)
Origin =       _Relationship("org", False, None, None, None, False)
Related =      _Relationship("rel", False, None, None, None, False)
Synonym =      _Relationship("syn", True,  0,    0,    0,    False)
Antonym =      _Relationship("ant", True,  0,    0,    0,    False)
NearSynonym =  _Relationship("nsy", True,  0,    0,    0,    False)
NearAntonym =  _Relationship("nat", True,  0,    0,    0,    False)
Euphemism =    _Relationship("eup", True,  0,    0,    0,    False)
Hyponym =      _Relationship("hpo", True,  1,    None, None, False)
Hypernym =     _Relationship("hpy", True,  -1,   None, None, False)
Cohyponym =    _Relationship("coo", True,  0,    None, None, False)
Holonym =      _Relationship("hol", True,  None, -1,   None, False)
Meronym =      _Relationship("mer", True,  None, 1,    None, False)
Comeronym =    _Relationship("com", True,  None, 0,    None, False)
Metonym =      _Relationship("met", True,  None, None, None, False)
Class =        _Relationship("cls", True,  None, None, -1,   False)
Instance =     _Relationship("ins", True,  None, None, 1,    False)
Abbreviation = _Relationship("abb", True,  0,    0,    0,    True)
Expansion =    _Relationship("exp", True,  0,    0,    0,    True)
Alternative =  _Relationship("alt", True,  0,    0,    0,    True)

# Define the symetric relationships
Compound.symetric =     Component
Component.symetric =    Compound
Derived.symetric =      Origin
Origin.symetric =       Derived
Related.symetric =      Related
Synonym.symetric =      Synonym
Antonym.symetric =      Antonym
NearSynonym.symetric =  NearSynonym
NearAntonym.symetric =  NearAntonym
Euphemism.symetric =    None
Hyponym.symetric =      Hypernym
Hypernym.symetric =     Hyponym
Cohyponym.symetric =    Cohyponym
Holonym.symetric =      Meronym
Meronym.symetric =      Holonym
Comeronym.symetric =    Comeronym
Metonym.symetric =      None
Class.symetric =        Instance
Instance.symetric =     Class
Abbreviation.symetric = Expansion
Expansion.symetric =    Abbreviation
Alternative.symetric =  Alternative



############## INITIALIZE PARTS OF SPEECH

POS_CODES = {}

@dataclass
class _PartOfSpeech:
	code:str            # Compact code that represents the POS
	isword:bool         # Whether this POS represents full words
	isatomic:bool       # Whether this POS is a unique element
	islexical:bool      # Whether this POS represents lexical elements (open class of semantically complete elements)
	isgrammatical: bool # Whether this POS represents grammatical elements (closed class of helper elements)

	def __post_init__(self):
		POS_CODES[self.code] = self

	def __eq__(self, pos):
		if isinstance(pos, str):
			return self.code == pos
		else:
			return self.code == pos.code

	def __ne__(self, pos):
		if isinstance(pos, str):
			return self.code != pos
		else:
			return self.code != pos.code

	def __add__(self, string):
		return self.code + string

	def __radd__(self, string):
		return string + self.code

	def __str__(self):
		return self.code

	def __hash__(self):
		return hash(self.code)

# Define the actual POS
Noun =                _PartOfSpeech("n",   True,  True,  True,  False)
ProperNoun =          _PartOfSpeech("pn",  True,  True,  True,  False)
Verb =                _PartOfSpeech("v",   True,  True,  True,  False)
PhrasalVerb =         _PartOfSpeech("phv", True,  False, True,  False)
Adjective =           _PartOfSpeech("adj", True,  True,  True,  False)
Numeral =             _PartOfSpeech("num", True,  True,  True,  False)
Adverb =              _PartOfSpeech("adv", True,  True,  True,  False)

Conjunction =         _PartOfSpeech("cnj", True,  True,  False, True)
Pronoun =             _PartOfSpeech("pro", True,  True,  False, True)
Article =             _PartOfSpeech("art", True,  True,  False, True)
Determiner =          _PartOfSpeech("det", True,  True,  False, True)
Preposition =         _PartOfSpeech("prp", True,  True,  False, True)
Postposition =        _PartOfSpeech("psp", True,  True,  False, True)
Interjection =        _PartOfSpeech("int", True,  True,  False, False)

Phrase =              _PartOfSpeech("phs", True,  False, True, False)
Proverb =             _PartOfSpeech("pvb", True,  False, True, False)
AdverbialPhrase =     _PartOfSpeech("aph", True,  False, True, False)
PrepositionalPhrase = _PartOfSpeech("ppp", True,  False, True, False)

Particle =            _PartOfSpeech("prt", False, True,  False, False)
Prefix =              _PartOfSpeech("pfx", False, True,  False, False)
Suffix =              _PartOfSpeech("sfx", False, True,  False, False)
Affix =               _PartOfSpeech("afx", False, True,  False, False)
Infix =               _PartOfSpeech("ifx", False, True,  False, False)
Interfix =            _PartOfSpeech("itx", False, True,  False, False)
Circumfix =           _PartOfSpeech("cfx", False, True,  False, False)

Symbol =              _PartOfSpeech("sym", False, True,  False, False)
Letter =              _PartOfSpeech("l",   False, True,  False, False)
Number =              _PartOfSpeech("nb",  False, True,  False, False)
DiacriticalMark =     _PartOfSpeech("dia", False, True,  False, False)

UnknownPOS =          _PartOfSpeech("unk", False, False, False, False)



############## WIKTIONARY PROCESSING INFORMATION

# Wiktionary section title -> POS
WIKTIONARY_POS = {
	"noun":         Noun,
	"plural noun":  Noun,
	"proper noun":  ProperNoun,
	"verb":         Verb,
	"phrasal verb": PhrasalVerb,
	"adjective":    Adjective,
	"numeral":      Numeral,
	"adverb":       Adverb,
	"conjunction":  Conjunction,
	"pronoun":      Pronoun,
	"article":      Article,
	"determiner":   Determiner,
	"preposition":  Preposition,
	"postposition": Postposition,
	"interjection": Interjection,
	"phrase":       Phrase,
	"proverb":      Proverb,
	"adverbial phrase":     AdverbialPhrase,
	"prepositional phrase": PrepositionalPhrase,
	"particle":     Particle,
	"prefix":       Prefix,
	"suffix":       Suffix,
	"affix":        Affix,
	"infix":        Infix,
	"interfix":     Interfix,
	"circumfix":    Circumfix,
	"symbol":       Symbol, 
	"letter":       Letter,
	"number":       Number,
	"diacritical mark":     DiacriticalMark,
	"unknown":      UnknownPOS,
}

# Wiktionary section title -> Relationship
RELATIONSHIP_SECTIONS = {
	"compounds":             Compound,
	"derived term":          Derived,
	"derived terms":         Derived,
	"derived terms=":        Derived,
	"obscure derivations":   Derived,
	"related":               Related,
	"related terms":         Related,
	"related words":         Related,
	"synonym":               Synonym,
	"synonyms":              Synonym,
	"synomyms":              Synonym,
	"opaque slang terms":    Synonym,
	"antonyms":              Antonym,
	"pseudo-synonyms":       NearSynonym,
	"near synonyms":         NearSynonym,
	"idiomatic synonyms":    NearSynonym,
	"ambiguous synonyms":    NearSynonym,
	"near antonyms":         NearAntonym,
	"euphemisms":            Euphemism,
	"hyponyms":              Hyponym,
	"troponyms":             Hyponym,
	"hypernym":              Hypernym,
	"hypernyms":             Hypernym,
	"coordinate terms":      Cohyponym,
	"coordinated terms":     Cohyponym,
	"cohyponyms":            Cohyponym,
	"holonyms":              Holonym,
	"meronyms":              Meronym,
	"comeronyms":            Comeronym,
	"metonyms":              Metonym,
	"class":                 Class,
	"classes":               Class,
	"instances":             Instance,
	"abbreviation":          Abbreviation,
	"abbreviations":         Abbreviation,
	"alternative form":      Alternative,
	"alternative forms":     Alternative,
	"alternate forms":       Alternative,
	"alternatie forms":      Alternative,
	"alternative spellings": Alternative,
	"alternative terms":     Alternative,
}


# Uninteresting sections to skip while parsing
SKIP_SECTIONS = (
	"english",         "etymology",      "pronunciation",
	"pronounciation",  "translations",   "see also",
	"references",      "usage notes",    "further reading",
	"anagrams",        "notes",          "trivia",
	"gallery",         "conjugation",    "quotations",
	"statistics",      "descendants",    "declension",
	"inflection",      "proverbs",       "punctuation mark",
	"contraction",     "paronyms",       "mutation",
	"various",         "similes",        "technical terms misused",
	"other",           "by type",        "archetypes",
	"usage",           "by degree",      "colloquialisms",
	"idioms",          "slang",          "colloquialisms or slang",
	"work to be done", "people",         "idioms/phrases",
	"pronucniation",   "collocations",   "adverb, preposition, conjunction",
	"further reasons", "proper names",   "colloquial, archaic, slang",
	"translation",     "race-based",     "race-based (warning- offensive)",
	"external links",  "quantification", "refeerences",
	"translatons",     "seealso",        "reference",
	"logogram",        "definition",     "animals",
	"plants",          "idiom",          "variance",
	"eymology",        "referenes",      "pronuciation",
	"initialisms",     "use",            "tranlations",
	"pronuncaition",   "by reason",      "by period of time",
	"demonyms",
)


# Uninteresting templates, to skip while parsing
SKIP_TEMPLATES = frozenset({"checksense", "top2", "mid2", "bottom", "i", "q", "qual", "qf", "qualifier", "s", "sense", "taxlink", "vern",
							"rel-top", "rel-mid", "rel-bottom", "der-top", "der-mid", "der-mid3", "der-bottom", "gloss",
							"rel-top3", "rel-mid3", "top3", "see derivation subpage", "lookfrom", "mid3", "g", "defdate",
							"rootsee", "mid", "prefixsee", "m", "m+", "rel-top4", "rel-mid4", "top4", "mid4", "der-top3", "affixes",
							"lb", "lbl", "label", "suffixsee", "-", "·", "c", "categorize", "topics", "top", "catlangname",
							"ipachar", "col-top", "hyp-top3", "hyp-mid3", "hyp-bottom", "affixes", "frac", "tea room",
							"misspelling", "zh-l", "initialism", "t+", "trans-top", "trans-mid", "trans-bottom", "syllable adjectives",
							"t", "pedia", "pedialite", "sense-lite", "top5", "mid5", "rfex", "isbn", "dated spelling of", "en-cont",
							"ux", "head", "bottom4", "en-contraction", "anchor", "wp", "desc", "hyp-top", "hyp-mid", "hyp-bottom",
							"attn", "attention", "bottom2", "t-needed", "der-top4", "der-mid4", "der top", "der bottom", "cog",
							"initialism of", "bottom3", "pedlink", "was wotd", "qualifier-lite", "en-conj", "decimate equivalents",
							"col-bottom", "decimation equivalents", "llc", "root", "sup", "cln", "punctuation", "prefixlanglemma",
							"comcatlite", "ws beginlist", "ws endlist", "ws ----", "lang", "gl", "syllable names", "arithmetic operations",
							"presidential nomics", "a", "syndiff", "bottom5", "rfi", "spaces", "suffix", "!", ",", "polyominoes", "unsupported",
							"seecites", "rfc", "wikispecies", "hu-verbpref", "common names of valerianella locusta", "gendered terms",
							"mediagenic terms", "alternative spelling of", "clear", "tlb", "trans-see", "n-g", "etyl", "interfixsee", "platyrrhini hypernyms",
							"cite-web", "iso 639", "circumfixsee", "sic", "non-gloss definition", "listen", "ja-l", "rfv-sense",
							"alternative form of", "vrrn", "cite book", "catlangcode", "australia", "hyp-top4"})

# Templates that all behave in the same way as lists of words
LIST_TEMPLATES = frozenset({"col", "col1", "col2", "col3", "col4", "col5", "col-u", "col1-u", "col2-u", "col3-u", "col4-u", "col5-u",
							"rel", "rel1", "rel2", "rel3", "rel4", "rel5", "rel-u", "rel1-u", "rel2-u", "rel3-u", "rel4-u", "rel5-u",
							"der", "der1", "der2", "der3", "der4", "der5", "der-u", "der1-u", "der2-u", "der3-u", "der4-u", "der5-u",
							"syn", "synonyms", "ant", "mer", "mero", "meronyms", "small", "collapse", "derived terms"})


class WiktionaryEntry (object):
	ATTR_WORD          = "w"
	ATTR_POS           = "pos"
	ATTR_RELATIONSHIPS = "rel"

	ID_SEPARATOR = "/"
	ID_SPACE = "_"

	def __init__(self, word, pos, relationships=None):
		self.word = word.lower()
		self.pos = pos
		self.id = None

		self.relationships = relationships or {}
		self.external_refs = set()

	def add_relationship(self, relationship, word):
		if relationship in self.relationships:
			self.relationships[relationship].add(word.lower())
		else:
			self.relationships[relationship] = {word.lower()}

	def add_relationships(self, relationship, words):
		if relationship in self.relationships:
			self.relationships[relationship].update(words)
		else:
			self.relationships[relationship] = set(words)

	def has_relationship(self, rel, word):
		if rel in self.relationships:
			if isinstance(word, WiktionaryEntry):
				return word.word in self.relationships[rel] or word.id in self.relationships[rel]
			else:
				return word.lower() in self.relationships[rel]
		else:
			return False

	@singledispatchmethod
	def merge(self, entry):
		self._merge_relationships(entry.relationships)

	@merge.register
	def _merge_relationships(self, relationships:dict):
		for relationship, words in relationships.items():
			self.add_relationships(relationship, words)

	def base_id(self):
		return self.word.replace(" ", self.ID_SPACE) + self.ID_SEPARATOR + self.pos.code + self.ID_SEPARATOR

	def save(self):
		out = {self.ATTR_WORD: self.word, self.ATTR_POS: self.pos.code,
				self.ATTR_RELATIONSHIPS: {rel.code: tuple(words) for rel, words in self.relationships.items()}}
		return out

	@classmethod
	def load(cls, id, dic):
		entry = cls(dic[cls.ATTR_WORD], POS_CODES[dic[cls.ATTR_POS]],
					{RELATIONSHIP_CODES[code]: set(words) for code, words in dic[cls.ATTR_RELATIONSHIPS].items()})
		entry.id = id
		return entry

	def __hash__(self):
		return hash(self.id)

	def __repr__(self):
		return f"{self.__class__.__name__}({repr(self.word)}, {repr(self.pos)}, {repr(self.relationships)})"
	
	def __str__(self):
		result = f"Wiktionary entry `{self.word}` as {self.pos}"
		for relationship, words in self.relationships.items():
			result += f"\n\t{relationship} : {', '.join(words)}"
		return result


	@classmethod
	def parse(cls, title, code):
		article = mw.parse(code)
		for heading in article.filter_headings():
			if heading.title.strip_code().lower() == "english":
				break
		else:
			return []

		entries = []

		languagelevel = None
		current_entry = None
		pos_level = None

		global_relationships = {}
		global_external_refs = set()

		for section in article.get_sections(flat=True, include_lead=False):
			heading = section.filter_headings()[0]
			heading_name = heading.title.strip_code().strip().lower()
			# Entering the english section
			if languagelevel is None and heading_name == "english":
				languagelevel = heading.level
			# We passed the english section heading
			elif languagelevel is not None:
				# Next language section : stop reading
				if heading.level <= languagelevel:
					break
				# Exiting the current part of speech
				elif pos_level is not None and heading.level <= pos_level and current_entry is not None:
					entries.append(current_entry)
					current_entry = None

				if heading_name.startswith(SKIP_SECTIONS) or heading_name == "":
					pass  # Uninteresting sections
				# Entering a new part of speech = a new entry
				elif heading_name in WIKTIONARY_POS:
					current_entry = WiktionaryEntry(title, WIKTIONARY_POS[heading_name])
					pos_level = heading.level
				elif current_entry is not None:
					if heading_name in RELATIONSHIP_SECTIONS:
						current_entry.add_relationships(RELATIONSHIP_SECTIONS[heading_name], cls.extract_terms(title, section, current_entry.external_refs))
					else:
						print(f"UNACCOUNTED SECTION {heading_name} in article {title}")
				elif heading_name in RELATIONSHIP_SECTIONS:
					terms = cls.extract_terms(title, section, global_external_refs)
					relationship = RELATIONSHIP_SECTIONS[heading_name]
					if relationship in global_relationships:
						global_relationships[relationship].update(terms)
					else:
						global_relationships[relationship] = set(terms)
				else:
					print(f"UNACCOUNTED SECTION OUT OF POS {heading_name} in article {title}")

		for entry in entries:
			entry.merge(global_relationships)
			entry.external_refs |= global_external_refs

		return entries

	@classmethod
	def extract_terms(cls, title, mwcode, external_refs):
		terms = set()
		#print("<-", mwcode.strip())

		next_is_ref = False
		nodes = [node for node in mwcode.nodes if node.strip().strip("*=,;|:") != ""]
		inbrackets = False
		for i, node in enumerate(nodes):
			if isinstance(node, (mw.nodes.heading.Heading, mw.nodes.comment.Comment, mw.nodes.html_entity.HTMLEntity, mw.nodes.external_link.ExternalLink)):
				continue
			elif isinstance(node, mw.nodes.text.Text):
				# See [[specific thesaurus page]] type of content
				if str(node).strip().lower() in ("see", "see also") and len(nodes) > i+1 and isinstance(nodes[i+1], (mw.nodes.wikilink.Wikilink, mw.nodes.tag.Tag)):
					next_is_ref = True
				else:
					content = str(node).strip()
					while "(" in content or ")" in content:
						leftindex = content.find("(")
						rightindex = content.find(")")
						if rightindex >= 0 and (rightindex < leftindex or leftindex < 0):  # Lone closing parenthesis
							content = content[rightindex+1:].strip()
							inbrackets = False
						elif rightindex < 0 and leftindex >= 0:  # Lone opening parenthesis
							content = content[:leftindex].strip()
							inbrackets = True
						else:  # Opening and closing parentheses in the right order
							content = (content[:leftindex] + content[rightindex+1:]).strip()

					for term in content.split(","):
						term = term.strip().strip("*\"'")
						if term != "":
							terms.add(term)
			elif not inbrackets:
				if isinstance(node, mw.nodes.wikilink.Wikilink):
					if next_is_ref:
						next_is_ref = False
						external_refs.add(str(node.title))
					elif str(node.title).startswith("Thesaurus:"):
						external_refs.add(str(node.title))
					elif not str(node.title).startswith("Category:"):
						terms.add(str(node.text if node.text is not None else node.title).strip())
				elif isinstance(node, mw.nodes.template.Template):
					nodeterms = cls.extract_template(title, node, external_refs)
					if len(nodeterms) > 0:
						terms.update(nodeterms)
				elif isinstance(node, mw.nodes.tag.Tag):
					# See [[specific thesaurus page]] type of content
					if str(node.contents).strip().lower() in ("see", "see also") and len(nodes) > i+1 and isinstance(nodes[i+1], (mw.nodes.wikilink.Wikilink, mw.nodes.tag.Tag)):
						next_is_ref = True
					elif next_is_ref and len(node.contents.nodes) > 0 and isinstance(node.contents.nodes[0], mw.nodes.wikilink.Wikilink):
							next_is_ref = False
							external_refs.add(str(node.contents.nodes[0].title))
					else:
						terms.update(cls.extract_terms(title, node.contents, external_refs))
				else:
					print("UNKNOWN NODE :", node, type(node))
		#print("->", terms, "\n\n")
		return {term.lower() for term in terms if "/" not in term and ":" not in term}

	@classmethod
	def extract_template(cls, title, template, external_refs):
		terms = set()
		name = template.name.strip().lower()
		params = cls.parse_template_parameters(template.params)
		if name in ("l", "ll", "link", "l-lite", "l-self", "topic"):
			if "3" in params:
				terms.update(cls.extract_terms(title, params["3"], external_refs))
			else:
				terms.update(cls.extract_terms(title, params["2"], external_refs))
		elif name == "w":
			if "2" in params:
				terms.update(cls.extract_terms(title, params["2"], external_refs))
			else:
				terms.update(cls.extract_terms(title, params["1"], external_refs))
		elif name in ("nowrap", "1", "smallcaps"):
			terms.update(cls.extract_terms(title, params["1"], external_refs))
		elif name == "ws":
			term = str(params["1"])
			terms.add(term)
			external_refs.add("Thesaurus:" + term)
		elif name in ("alt", "alter"):
			for i in cls.param_range(2):
				if i in params:
					if str(params[i]).strip() == "":
						break
					else:
						terms.update(cls.extract_terms(title, params[i], external_refs))
				else:
					break
		elif name == "seesynonyms":
			if "2" not in params:
				external_refs.add("Thesaurus:" + title)
			else:
				for i in cls.param_range(2):
					if i in params:
						external_refs.add("Thesaurus:" + str(params[i]))
					else:
						break
		elif name in ("see", "also", "see also"):
			for i in cls.param_range(1):
				if i in params:
					if params[i].startswith("Thesaurus:"):
						external_refs.add(str(params[i]))
				else:
					break
		elif name == "examples":
			if "examples" in params:
				exlist = params["examples"]
				terms.update(cls.extract_terms(title, exlist, external_refs))
			elif "example" in params:
				exlist = params["example"]
				terms.update(cls.extract_terms(title, exlist, external_refs))
		elif name.startswith("wikisaurus:"):
			external_refs.add("Thesaurus:" + name.partition(":")[2])
		elif name in LIST_TEMPLATES:
			for i in cls.param_range(2):
				if i in params:
					terms.update(cls.extract_terms(title, params[i], external_refs))
				else:
					break
		elif name in SKIP_TEMPLATES or len(params) == 0:
			pass  # Uninteresting templates
			# FIXME : taxonomic links …?
		elif name.startswith(("r:", "rq:", "list:", "webster", "wikipedia", "cite-journal", "quote-book", "cite-book", "template:", "table:", "projectlink")):
			pass  # Reference templates
		else:
			print(f"\nUNKNOWN TEMPLATE {name} in article {title} : {template}")
		return terms

	@classmethod
	def parse_template_parameters(self, params):
		parsed = {}
		for param in params:
			parsed[str(param.name)] = param.value
		return parsed

	@classmethod
	def param_range(cls, start=0, step=1):
		i = start
		while True:
			yield str(i)
			i += step


class Wiktionary (object):
	def __init__(self):
		self.word_entries = {}
		self.pos_entries = {}
		self.entries = {}

		self.word_thesaurus = {}
	
	def add(self, entry:WiktionaryEntry, setid=True):
		"""Add a new Wiktionary entry.
		   Default behaviour is to automatically set the entry id, setid=False to disable it"""
		if entry.word in self.word_entries:
			self.word_entries[entry.word].add(entry)
		else:
			self.word_entries[entry.word] = {entry}
		
		wordpos = (entry.word, entry.pos)
		if wordpos in self.pos_entries:
			if setid:
				entry.id = entry.base_id() + str(len(self.pos_entries[wordpos]))
			self.pos_entries[wordpos].add(entry)
		else:
			if setid:
				entry.id = entry.base_id() + "0"
			self.pos_entries[wordpos] = {entry}

		self.entries[entry.id] = entry

	def add_thesaurus(self, entry:WiktionaryEntry):
		if entry.word in self.word_thesaurus:
			self.word_thesaurus[entry.word].add(entry)
		else:
			self.word_thesaurus[entry.word] = {entry}
	
	def add_all(self, entries:Iterable[WiktionaryEntry]):
		for entry in entries:
			if entry.word.lower().startswith("thesaurus:"):
				self.add_thesaurus(entry)
			else:
				self.add(entry)

	def merge_references(self):
		for i, (id, entry) in enumerate(self.entries.items()):
			for reference in entry.external_refs:
				if reference in self.word_thesaurus:
					thesaurus_entries = self.word_thesaurus[reference]
					for thentry in thesaurus_entries:
						if thentry.pos == entry.pos:
							entry.merge(thentry)
			if i % 1000 == 0:
				print(f"\rProcessed {i+1}/{len(self.entries)}", end="")
		print(f"\rProcessed {i+1}/{len(self.entries)}")

	def preprocess(self):
		for i, entry in enumerate(set(self.entries.values())):
			for rel, relset in tuple(entry.relationships.items()):
				entry.relationships[rel] = self._resolve_relationships(entry, rel, relset)

			if i % 1000 == 0:
				print(f"\rProcessed {i+1}/{len(self.entries)}", end="")
		print(f"\rProcessed {i+1}/{len(self.entries)}")

	def _related_entries(self, entry, rel, relset):
		entries = []
		for relword in relset:
			if self.isid(relword):
				if relword not in self.entries:
					word, pos, index = relword.split(WiktionaryEntry.ID_SEPARATOR)
					dummy = WiktionaryEntry(word, POS_CODES[pos])
					dummy.id = relword
					self.add(dummy, setid=False)
				entries.append(self.entries[relword])
			else:
				if rel.samepos:
					if (relword, entry.pos) not in self.pos_entries:
						self.add(WiktionaryEntry(relword, entry.pos), setid=True)
					lookup = self.pos_entries[(relword, entry.pos)]
				else:
					if relword not in self.word_entries:
						self.add(WiktionaryEntry(relword, UnknownPOS), setid=True)
					lookup = self.word_entries[relword]

				found = []
				for lookupentry in lookup:
					if lookupentry.has_relationship(rel.symetric, entry):
						found.append(lookupentry)

				if len(found) > 0:
					entries.extend(found)
				else:
					entries.extend(lookup)
		return entries

	def _resolve_relationships(self, entry, rel, relset):
		related = self._related_entries(entry, rel, relset)
		newrelset = set()
		for relentry in related:
			if ":" not in relentry.word:
				newrelset.add(relentry.id)
				if rel.symetric is not None:
					relentry.add_relationship(rel.symetric, entry.id)
		return newrelset



	def save(self, filename):
		output = {id: entry.save() for id, entry in self.entries.items()}
		with open(filename, "w", encoding="utf-8") as jsonfile:
			json.dump(output, jsonfile, indent=4)

	@classmethod
	def isid(self, word):
		return word.count(WiktionaryEntry.ID_SEPARATOR) == 2

	@classmethod
	def load(cls, filename):
		with open(filename, "r", encoding="utf-8") as jsonfile:
			serialized = json.load(jsonfile)

		wiktionary = cls()
		for id, entry in serialized.items():
			wiktionary.add(WiktionaryEntry.load(id, entry), setid=False)
		return wiktionary


	def __len__(self):
		return len(self.entries)

	def __contains__(self, word):
		return word in self.word_entries


class WiktionaryHandler (ContentHandler):
	def __init__(self):
		self.dictionary = Wiktionary()

	def startDocument(self):
		self.read_pages = 0
		self.processed_pages = 0
		self.processed_entries = 0
		self.input_title = False
		self.input_text = False
		self.current_title = ""
		self.current_text = ""
    
	def startElement(self, name, attrs):
		if name == "title":
			self.input_title = True
			self.current_title = ""
		elif name == "text":
			self.input_text = True
			self.current_text = ""
	
	def endElement(self, name):
		if name == "title":
			self.input_title = False
		elif name == "text":
			self.input_text = False
			self.read_pages += 1
			if (self.current_title.lower().startswith("thesaurus:") or ":" not in self.current_title) and "/" not in self.current_title:
				entries = WiktionaryEntry.parse(self.current_title, self.current_text)
				if len(entries) > 0:
					self.dictionary.add_all(entries)
					self.processed_pages += 1
					self.processed_entries += len(entries)
			if self.read_pages % 100 == 0:
				print(f"\rProcessed {self.processed_entries}/{self.processed_pages}/{self.read_pages}", end="")
	
	def characters(self, content):
		if self.input_title:
			self.current_title += content
		elif self.input_text:
			self.current_text += content


if __name__ == "__main__":
	handler = WiktionaryHandler()
	try:
		with open("enwiktionary.xml", "r", encoding="utf-8") as f:
			xml.sax.parse(f, handler)
	except KeyboardInterrupt:
		pass
	print("\nMerging references")
	handler.dictionary.merge_references()
	print("Preprocessing")
	handler.dictionary.preprocess()
	print("Saving")
	handler.dictionary.save("wiktionary.json")
