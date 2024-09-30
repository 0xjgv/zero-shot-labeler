# Named Entity Recognition (NER) Handler
from playground.methods.aho_corasick import AhoCorasick


def ner(
    *,
    patterns: list[dict[str, str]],
    fuzzy_threshold: int = 0,
    message_subject: str,
    message_body: str,
) -> list[dict[str, str]]:
    matches = []
    if len(patterns) > 0:
        aho_corasick_ner = AhoCorasick(patterns=patterns)
        aho_corasick_results_subject = aho_corasick_ner(
            text=message_subject, **{"is_subject": True}
        )
        aho_corasick_results_body = aho_corasick_ner(
            text=message_body, **{"is_body": True}
        )
        matches.extend(aho_corasick_results_subject["matches"])
        matches.extend(aho_corasick_results_body["matches"])

    return matches
