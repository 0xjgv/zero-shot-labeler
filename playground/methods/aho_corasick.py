# Named Entity Recognition (NER) Handler
from typing import Any

import ahocorasick


class TrieNode:
    __slots__ = ["children", "outputs", "fail"]

    def __init__(self):
        self.children = {}
        self.outputs = []
        self.fail = None


class AhoCorasickNER:
    __slots__ = ["automaton", "patterns"]

    def __init__(self, *, patterns: list[dict[str, str]]):
        self.automaton = ahocorasick.Automaton()
        self.patterns = patterns

        for pattern in patterns:
            for key, value in pattern.items():
                if key != "_id":
                    self.automaton.add_word(
                        self.normalize(value), (pattern["_id"], key, value)
                    )

        self.automaton.make_automaton()

    def normalize(self, text: str) -> str:
        return text.lower()

    def __call__(self, *, text: str) -> dict[str, Any]:
        normalized_text = self.normalize(text)
        matches = []

        for i, data in self.automaton.iter(normalized_text):
            matches.append(
                {
                    "start_index": i - len(data[2]) + 1,
                    "end_index": i + 1,
                    "value": data[2],
                    "key": data[1],
                    "_id": data[0],
                }
            )
        return {"matches": matches}


if __name__ == "__main__":
    text = "Hello, my name is JOHN DOE. I live in NEW YORK CITY. My contract number is co-3456. OP-1234 and the order or-2345."
    targets = [
        {"_id": "3", "opportunity_number": "OP-1234", "_title": "Opportunity OP-1234"},
        {"_id": "2", "contract_number": "CO-3456", "_title": "Contract CO-3456"},
        {"_id": "1", "order_number": "OR-2345", "_title": "Order OR-2345"},
    ]

    ner = AhoCorasickNER(patterns=targets)
    matches = ner.process_text(text)
    for m in matches["matches"]:
        print(m)
