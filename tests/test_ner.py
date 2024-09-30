import pytest

from playground.ner import ner

# Test data
sample_text = "Hello, my name is JOHN DOE. I live in NEW YORK CITY. My contract number is co-3456. OP-1234 and the order or-2345."
sample_patterns = [
    {"_id": "3", "opportunity_number": "OP-1234", "_title": "Opportunity OP-1234"},
    {"_id": "2", "contract_number": "CO-3456", "_title": "Contract CO-3456"},
    {"_id": "1", "order_number": "OR-2345", "_title": "Order OR-2345"},
]


def test_ner_function_basic():
    matches = ner(
        message_subject="Sample subject",
        message_body=sample_text,
        patterns=sample_patterns,
    )

    expected_matches = [
        {
            "key": "contract_number",
            "start_index": 75,
            "value": "CO-3456",
            "end_index": 82,
            "is_body": True,
            "_id": "2",
        },
        {
            "key": "opportunity_number",
            "value": "OP-1234",
            "start_index": 84,
            "end_index": 91,
            "is_body": True,
            "_id": "3",
        },
        {
            "value": "Order OR-2345",
            "start_index": 100,
            "end_index": 113,
            "is_body": True,
            "key": "_title",
            "_id": "1",
        },
        {
            "key": "order_number",
            "start_index": 106,
            "value": "OR-2345",
            "end_index": 113,
            "is_body": True,
            "_id": "1",
        },
    ]

    assert matches == expected_matches


def test_ner_function_case_insensitive():
    lowercase_text = sample_text.lower()

    matches = ner(
        message_subject="Sample subject",
        message_body=lowercase_text,
        patterns=sample_patterns,
    )

    assert len(matches) == 4


def test_ner_function_no_matches():
    no_match_text = "This text contains no matching patterns."
    matches = ner(
        message_subject="No matches here",
        message_body=no_match_text,
        patterns=sample_patterns,
    )
    assert len(matches) == 0


def test_ner_function_overlapping_patterns():
    overlapping_patterns = [
        {"_id": "1", "pattern": "abc"},
        {"_id": "2", "pattern": "bcd"},
    ]
    text = "abcd"

    matches = ner(
        patterns=overlapping_patterns,
        message_subject="Sample subject",
        message_body=text,
    )

    assert matches[0]["value"] == "abc"
    assert matches[1]["value"] == "bcd"
    assert len(matches) == 2


def test_ner_function_empty_input():
    matches = ner(
        patterns=sample_patterns,
        message_subject="",
        message_body="",
    )
    assert len(matches) == 0


def test_ner_function_empty_patterns():
    matches = ner(
        message_subject="Sample subject",
        message_body="",
        patterns=[],
    )

    assert len(matches) == 0


def test_ner_function_multiple_occurrences():
    text = "OP-1234 is an opportunity. Another opportunity is OP-1234."
    patterns = [{"_id": "1", "opportunity_number": "OP-1234"}]

    matches = ner(
        message_subject="Sample subject",
        message_body=text,
        patterns=patterns,
    )

    assert matches[1]["start_index"] == 50
    assert matches[0]["start_index"] == 0
    assert len(matches) == 2


def test_ner_function_pattern_at_text_boundaries():
    patterns = [{"_id": "1", "opportunity_number": "OP-1234"}]
    text = "OP-1234 is at the start and the end is OP-1234"

    matches = ner(
        message_subject="Sample subject",
        message_body=text,
        patterns=patterns,
    )

    assert matches[1]["start_index"] == 39
    assert matches[0]["start_index"] == 0
    assert len(matches) == 2


def test_ner_function_with_special_characters():
    patterns = [{"_id": "1", "opportunity_number": "OP-1234"}]
    text = "Special chars: !@#$%^&*() and OP-1234."
    matches = ner(
        message_subject="Sample subject",
        message_body=text,
        patterns=patterns,
    )

    assert matches[0]["start_index"] == 30
    assert len(matches) == 1


@pytest.mark.parametrize(
    "input_text,expected_count",
    [
        ("Three matches: OP-1234, CO-3456, and OR-2345", 3),
        ("Two matches: OP-1234 and CO-3456", 2),
        ("One match: OP-1234", 1),
        ("No matches here", 0),
        ("", 0),
    ],
)
def test_ner_function_parametrized(input_text, expected_count):
    matches = ner(
        message_subject="Sample subject",
        message_body=input_text,
        patterns=sample_patterns,
    )
    assert len(matches) == expected_count


def test_ner_function_german_email():
    message_subject = (
        "Betreff: Auftragsstatus AUF-2023-001 und Vertragsverlängerung VTR-4567"
    )
    message_body = """
    Betreff: Auftragsstatus AUF-2023-001 und Vertragsverlängerung VTR-4567

    Sehr geehrter Herr Müller,

    ich hoffe, diese E-Mail erreicht Sie gut. Ich möchte Sie über den aktuellen Stand des Auftrags AUF-2023-001 informieren und die anstehende Vertragsverlängerung VTR-4567 besprechen.

    1. Auftragsstatus AUF-2023-001:
       Der Auftrag befindet sich in der finalen Phase. Unser Team arbeitet hart daran, ihn bis zum 15. Mai abzuschließen.

    2. Vertragsverlängerung VTR-4567:
       Ihr aktueller Vertrag VTR-4567 läuft am 30. Juni aus. Wir möchten Ihnen eine Verlängerung zu verbesserten Konditionen anbieten.

    3. Neue Gelegenheit GEL-789:
       Außerdem möchte ich Sie auf eine neue Geschäftsmöglichkeit mit der Kennung GEL-789 aufmerksam machen.

    Bei Fragen stehe ich Ihnen gerne zur Verfügung. Sie erreichen mich unter der Durchwahl DW-123.

    Mit freundlichen Grüßen,
    Max Mustermann
    Kundenbetreuer
    """

    german_patterns = [
        {"_id": "2", "vertragsnummer": "VTR-4567", "_title": "Vertrag VTR-4567"},
        {"_id": "3", "gelegenheit": "GEL-789", "_title": "Gelegenheit GEL-789"},
        {"_id": "4", "durchwahl": "DW-123", "_title": "Durchwahl DW-123"},
        {
            "auftragsnummer": "AUF-2023-001",
            "_title": "Auftrag AUF-2023-001",
            "_id": "1",
        },
    ]

    matches = ner(
        message_subject=message_subject,
        message_body=message_body,
        patterns=german_patterns,
    )

    assert len(matches) == 15

    # Check if all expected patterns are found
    found_patterns = set(match["value"] for match in matches)
    expected_patterns = {
        # "Auftrag AUF-2023-001",
        "Gelegenheit GEL-789",
        "Durchwahl DW-123",
        "Vertrag VTR-4567",
        "AUF-2023-001",
        "VTR-4567",
        "GEL-789",
        "DW-123",
    }
    assert found_patterns == expected_patterns

    # Check if the matches are in the correct order
    match_values = [match["value"] for match in matches]
    assert (
        match_values.index("AUF-2023-001")
        < match_values.index("VTR-4567")
        < match_values.index("GEL-789")
        < match_values.index("DW-123")
    )

    # Check if the start and end indices are correct for one of the matches
    auf_match = next(match for match in matches if match["value"] == "AUF-2023-001")
    assert (
        message_subject[auf_match["start_index"] : auf_match["end_index"]]
        == "AUF-2023-001"
    )

    # Check if case-insensitive matching works
    lowercase_subject = message_subject.lower()
    lowercase_body = message_body.lower()
    lowercase_matches = ner(
        message_subject=lowercase_subject,
        message_body=lowercase_body,
        patterns=german_patterns,
    )
    assert len(lowercase_matches) == len(matches)


def test_ner_function_german_email_with_fuzzy_matching():
    message_subject = (
        "Betreff: Auftragsstatus AUF-2023-001 und Vertragsverlängerung VTR-4567"
    )
    message_body = """
    Betreff: Auftragsstatus AUF-2023-001 und Vertragsverlängerung VTR-4567

    Sehr geehrter Herr Müller,

    ich hoffe, diese E-Mail erreicht Sie gut. Ich möchte Sie über den aktuellen Stand des Auftrags AUF-2023-001 informieren und die anstehende Vertragsverlängerung VTR-4567 besprechen.

    1. Auftragsstatus AUF-2023-001:
       Der Auftrag befindet sich in der finalen Phase. Unser Team arbeitet hart daran, ihn bis zum 15. Mai abzuschließen.

    2. Vertragsverlängerung VTR-4567:
       Ihr aktueller Vertrag VTR-4567 läuft am 30. Juni aus. Wir möchten Ihnen eine Verlängerung zu verbesserten Konditionen anbieten.

    3. Neue Gelegenheit GEL-789:
       Außerdem möchte ich Sie auf eine neue Geschäftsmöglichkeit mit der Kennung GEL-789 aufmerksam machen.

    Bei Fragen stehe ich Ihnen gerne zur Verfügung. Sie erreichen mich unter der Durchwahl DW-123.

    Mit freundlichen Grüßen,
    Max Mustermann
    Kundenbetreuer
    """

    german_patterns = [
        {"_id": "2", "vertragsnummer": "VTR-4567", "_title": "Vertrag VTR-4567"},
        {"_id": "3", "gelegenheit": "GEL-789", "_title": "Gelegenheit GEL-789"},
        {"_id": "4", "durchwahl": "DW-123", "_title": "Durchwahl DW-123"},
        {
            "auftragsnummer": "AUF-2023-001",
            "_title": "Auftrag AUF-2023-001",
            "_id": "1",
        },
    ]

    # Use fuzzy matching with a threshold of 1
    matches = ner(
        message_subject=message_subject,
        message_body=message_body,
        patterns=german_patterns,
        fuzzy_threshold=1,
    )

    assert len(matches) == 15

    # Check if all expected patterns are found
    found_patterns = set(match["value"] for match in matches)
    expected_patterns = {
        # "Auftrag AUF-2023-001",
        "Gelegenheit GEL-789",
        "Durchwahl DW-123",
        "Vertrag VTR-4567",
        "AUF-2023-001",
        "VTR-4567",
        "GEL-789",
        "DW-123",
    }
    assert found_patterns == expected_patterns

    # Check if the matches are in the correct order
    match_values = [match["value"] for match in matches]
    assert (
        match_values.index("AUF-2023-001")
        < match_values.index("VTR-4567")
        < match_values.index("GEL-789")
        < match_values.index("DW-123")
    )

    # Check if the start and end indices are correct for one of the matches
    auf_match = next(match for match in matches if match["value"] == "AUF-2023-001")
    assert (
        message_subject[auf_match["start_index"] : auf_match["end_index"]]
        == "AUF-2023-001"
    )

    # Test fuzzy matching with a slight typo
    message_body_with_typo = message_body.replace("AUF-2023-001", "AUF2023-00l")
    matches_with_typo = ner(
        message_body=message_body_with_typo,
        message_subject=message_subject,
        patterns=german_patterns,
        fuzzy_threshold=1,
    )
    assert len(matches_with_typo) == 12


# Add more tests as needed
