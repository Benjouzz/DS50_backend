import unicodedata
from functools import singledispatchmethod

from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords


INVERTER_WORDS = {"no", "non", "not", "anti"}
STOPWORDS = set(stopwords.words("english")) - INVERTER_WORDS
INITIAL_BANWORDS = {"read", "reading", "own", "my", "default", "meine", "favorite", "favourite", "kindle"}
ENGLISH_WORDS = set(wordnet.words())

LEMMATIZER = WordNetLemmatizer()


def is_latin(text):
	for char in text:
		if char.isalpha():
			if not unicodedata.name(char).startswith("LATIN"):
				return False
	return True


class Name (object):
	NLTK_WORDNET_POS_CONVERSION = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

	def __init__(self, name):
		if isinstance(name, str):
			tokens = Name.parse(name)
			self._tokens = frozenset(tokens)
		elif isinstance(name, (set, frozenset)):
			self._tokens = frozenset(name)
		self._words = frozenset({word for word, pos, hint in self._tokens})
		self._posmap = {word: pos for word, pos, hint in self._tokens}


	@staticmethod
	def parse(textname):
		words = Name.tokenize(textname)
		tokens = set()
		for word, pos in pos_tag(words):
			if word not in STOPWORDS:
				pos = Name.convert_pos(pos)
				lemma = LEMMATIZER.lemmatize(word, pos=pos)
				if lemma in ENGLISH_WORDS:
					existing_pos = [synset.pos() for synset in wordnet.synsets(lemma)]
					if pos not in existing_pos:
						if len(existing_pos) > 0:
							pos = existing_pos[0]
						else:
							continue
				tokens.add((lemma, pos, None))
		return tokens

	@staticmethod
	def convert_pos(pos):
		if pos[0] in Name.NLTK_WORDNET_POS_CONVERSION:
			return Name.NLTK_WORDNET_POS_CONVERSION[pos[0]]
		else:
			return wordnet.NOUN

	@staticmethod
	def isalpha(string):
		return all([char.isalpha() or char == "_" for char in string])

	@staticmethod
	def tokenize(textname):
		tokens = [""]
		for char in textname.lower():
			if Name.isalpha(char):
				if Name.isalpha(tokens[-1]):
					tokens[-1] += char
				else:
					tokens.append(char)
			elif char.isdigit():
				if tokens[-1].isdigit():
					tokens[-1] += char
				else:
					tokens.append(char)
			elif char in "- ":
				tokens.append("")
		return [token.strip() for token in tokens if len(token) > 0]

	def tokens(self):
		return self._tokens

	def words(self):
		return self._words

	def pos(self, word):
		return self._posmap[word]

	def token(self, word):
		return (word, self._posmap[word])

	def tostring(self, withhint=False):
		return "+".join([
			word + ":" + pos + (":" + hint if withhint and hint is not None else "")
			for word, pos, hint in self._tokens])

	@classmethod
	def fromstring(cls, string):
		tokens = set()
		for item in string.split("+"):
			elements = item.split(":")
			tokens.add((elements[0], elements[1], elements[2] if len(elements) > 2 else None))
		return cls(tokens)

	def _gettokens(self, obj):
		if isinstance(obj, Name):
			return obj._tokens
		else:
			return set(obj)

	def _getwords(self, obj):
		if isinstance(obj, Name):
			return obj._words
		else:
			return set(obj)

	def __and__(self, other):
		return Name(self._tokens & self._gettokens(other))
	def __or__(self, other):
		return Name(self._tokens | self._gettokens(other))
	def __xor__(self, other):
		return Name(self._tokens ^ self._gettokens(other))
	def __sub__(self, other):
		return Name(self._tokens - self._gettokens(other))

	def __eq__(self, other):
		return self._words == self._getwords(other)
	def __ne__(self, other):
		return self._words != self._getwords(other)
	def __gt__(self, other):
		return self._words > self._getwords(other)
	def __ge__(self, other):
		return self._words >= self._getwords(other)
	def __lt__(self, other):
		return self._words < self._getwords(other)
	def __le__(self, other):
		return self._words <= self._getwords(other)

	def __rand__(self, other):
		return self._getwords(other) & self._words
	def __ror__(self, other):
		return self._getwords(other) | self._words
	def __rxor__(self, other):
		return self._getwords(other) ^ self._words
	def __rsub__(self, other):
		return self._getwords(other) - self._words

	def __len__(self):
		return len(self._tokens)

	def __iter__(self):
		return iter(self._tokens)

	def __contains__(self, item):
		if isinstance(item, str):
			return item in self._words
		else:
			return item in self._tokens

	def __hash__(self):
		return hash(self._words)

	def __str__(self):
		return self.tostring()

	def __repr__(self):
		return repr(self._tokens)


class HierarchyItem (object):
	def __init__(self, id, title, names, superclass=None, subclasses=[], level=0, path=[], favorite_select=False):
		self.id = id
		self.title = title
		self.names = list(names)
		self.superclass = superclass
		self.subclasses = list(subclasses)
		self.level = level
		self.path = list(path)
		self.favorite_select = favorite_select

	@classmethod
	def load(cls, id, jsonitem):
		return cls(id, jsonitem["title"], [Name.fromstring(name) for name in jsonitem["names"]], jsonitem["sup"], jsonitem["sub"], jsonitem["lvl"], jsonitem["path"], jsonitem["fav"])

	def save(self):
		return {"title": self.title, "names": [name.tostring(withhint=True) for name in self.names], "sup": self.superclass, "sub": self.subclasses, "lvl": self.level, "path": self.path, "fav": self.favorite_select}


class Hierarchy (object):
	def __init__(self):
		self.entries = {}
		self.names = {}

	@classmethod
	def load(cls, jsondic):
		self = cls()
		self.entries = {int(id): HierarchyItem.load(int(id), entry) for id, entry in jsondic.items()}
		self.names = {}
		self.max_id = max(self.entries.keys())
		for id, entry in self.entries.items():
			for name in entry.names:
				self.names[name] = id
		return self

	def save(self):
		return {id: entry.save() for id, entry in self.entries.items()}

	def add(self, id, entry):
		self.entries[id] = entry
		for name in entry.names:
			self.names[name] = id

	def remove(self, id):
		entry = self.entries[id]
		for name in entry.names:
			del self.names[name]
		del self.entries[id]

	def move(self, old_id, new_id):
		entry = self.entries[old_id]
		if new_id in self.entries:
			self.entries[old_id] = self.entries[new_id]
			self.entries[new_id] = entry
		else:
			self.entries[new_id] = entry
			del self.entries[old_id]

	def maxid(self):
		return max(self.entries.keys())

	@singledispatchmethod
	def get(self, id):
		raise NotImplementedError()

	@get.register
	def _get_id(self, id:int):
		return self.entries[id]

	@get.register
	def _get_name(self, name:Name):
		return self.entries[self.names[name]]

	def get_title(self, title:str):
		for entry in self.entries.values():
			if entry.title == title:
				return entry

	def __len__(self):
		return len(self.entries)

	def __iter__(self):
		return iter(self.entries.values())