# Named Entity Recognition (NER) Handler
from playground.methods.aho_corasick import AhoCorasickNER


def ner(
    *,
    patterns: list[dict[str, str]],
    fuzzy_threshold: int = 0,
    text: str,
) -> list[dict[str, str]]:
    matches = []
    if len(patterns) > 0:
        aho_corasick_ner = AhoCorasickNER(patterns=patterns)
        aho_corasick_results = aho_corasick_ner(text=text)
        matches.extend(aho_corasick_results["matches"])
    return matches
