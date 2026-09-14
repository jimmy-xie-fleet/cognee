from unittest.mock import MagicMock

from cognee.modules.engine.models.Assertion import Assertion, verify_source_quote
from cognee.modules.engine.models.Entity import Entity


class TestAssertionIdentity:
    def test_auto_derived_id_matches_id_for_in_identity_field_order(self):
        assertion = Assertion(
            name="the sky is blue",
            description="a claim about the sky",
            source_chunk_id="chunk-1",
            statement_type="testimony",
            asserted_by="witness a",
            occurrence=2,
        )
        expected_id = Assertion.id_for("the sky is blue", "chunk-1", "testimony", "witness a", 2)
        assert assertion.id == expected_id

    def test_differs_from_entity_with_same_name_and_description(self):
        assertion = Assertion(name="x", description="d", statement_type="denial")
        entity = Entity(name="x", description="d")
        assert assertion.id != entity.id

    def test_changing_only_statement_type_changes_id(self):
        base = Assertion(
            name="x",
            description="d",
            statement_type="allegation",
            asserted_by="a",
            source_chunk_id="c1",
            occurrence=1,
        )
        changed = Assertion(
            name="x",
            description="d",
            statement_type="denial",
            asserted_by="a",
            source_chunk_id="c1",
            occurrence=1,
        )
        assert base.id != changed.id

    def test_changing_only_asserted_by_changes_id(self):
        base = Assertion(
            name="x",
            description="d",
            statement_type="allegation",
            asserted_by="a",
            source_chunk_id="c1",
            occurrence=1,
        )
        changed = Assertion(
            name="x",
            description="d",
            statement_type="allegation",
            asserted_by="b",
            source_chunk_id="c1",
            occurrence=1,
        )
        assert base.id != changed.id

    def test_changing_only_source_chunk_id_changes_id(self):
        base = Assertion(
            name="x",
            description="d",
            statement_type="allegation",
            asserted_by="a",
            source_chunk_id="c1",
            occurrence=1,
        )
        changed = Assertion(
            name="x",
            description="d",
            statement_type="allegation",
            asserted_by="a",
            source_chunk_id="c2",
            occurrence=1,
        )
        assert base.id != changed.id

    def test_changing_only_occurrence_changes_id(self):
        base = Assertion(
            name="x",
            description="d",
            statement_type="allegation",
            asserted_by="a",
            source_chunk_id="c1",
            occurrence=1,
        )
        changed = Assertion(
            name="x",
            description="d",
            statement_type="allegation",
            asserted_by="a",
            source_chunk_id="c1",
            occurrence=2,
        )
        assert base.id != changed.id

    def test_type_attribute_is_assertion(self):
        assertion = Assertion(name="x", description="d", statement_type="statement")
        assert assertion.type == "Assertion"

    def test_valid_to_defaults_none_and_not_identity_field(self):
        assertion = Assertion(name="x", description="d", statement_type="statement")
        assert assertion.valid_to is None

        with_valid_to = Assertion(
            name="x",
            description="d",
            statement_type="statement",
            valid_to=123,
        )
        assert with_valid_to.id == assertion.id


class TestVerifySourceQuote:
    def test_exact_substring_match(self):
        assert verify_source_quote("hello world", "say hello world today") is True

    def test_curly_quotes_and_line_wrapped_whitespace_match(self):
        quote = "the party’s “final” offer\nwas rejected"
        text = 'Earlier, the party\'s "final" offer was rejected outright.'
        assert verify_source_quote(quote, text) is True

    def test_different_casing_matches(self):
        assert verify_source_quote("HELLO world", "hello WORLD") is True

    def test_missing_quote_returns_false(self):
        assert verify_source_quote("not present anywhere", "some other text") is False

    def test_none_quote_returns_false(self):
        assert verify_source_quote(None, "some text") is False

    def test_empty_quote_returns_false(self):
        assert verify_source_quote("", "some text") is False

    def test_magicmock_text_returns_false(self):
        assert verify_source_quote("hello", MagicMock()) is False

    def test_none_text_returns_false(self):
        assert verify_source_quote("hello", None) is False
