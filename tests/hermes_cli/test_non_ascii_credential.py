"""Tests for invalid-header-character credential detection and sanitization.

Covers two pollution patterns that produce opaque HTTP 400s from the
provider before the request body is even parsed:

- Unicode lookalikes (e.g. ʋ U+028B instead of v) from copy-pasting
  keys through PDFs or rich-text editors. Original report: issue #6843.
- ASCII control bytes such as ESC (0x1B) captured when typing keys
  interactively and using arrow keys, which embed terminal control
  sequences directly into the value. Both classes of bytes are invalid
  in HTTP header values per RFC 7230 (visible USASCII + space only).
"""

import os

from hermes_cli.config import _check_non_ascii_credential


class TestCheckNonAsciiCredential:
    """Tests for _check_non_ascii_credential()."""

    def test_ascii_key_unchanged(self):
        key = "sk-proj-" + "a" * 100
        result = _check_non_ascii_credential("TEST_API_KEY", key)
        assert result == key

    def test_strips_unicode_v_lookalike(self, capsys):
        """The exact scenario from issue #6843: ʋ instead of v."""
        key = "sk-proj-abc" + "ʋ" + "def"  # \u028b
        result = _check_non_ascii_credential("OPENROUTER_API_KEY", key)
        assert result == "sk-proj-abcdef"
        assert "ʋ" not in result
        # Should print a warning
        captured = capsys.readouterr()
        assert "break API requests" in captured.err

    def test_strips_multiple_non_ascii(self, capsys):
        key = "sk-proj-aʋbécd"
        result = _check_non_ascii_credential("OPENAI_API_KEY", key)
        assert result == "sk-proj-abcd"
        captured = capsys.readouterr()
        assert "U+028B" in captured.err  # reports the char

    def test_strips_ansi_escape_from_interactive_paste(self, capsys):
        """Arrow keys captured during interactive setup embed ESC sequences.

        Real-world repro: user pressed Up/Left arrow inside the setup
        prompt before pasting the key, capturing ``\\x1b[A`` (up) and
        ``\\x1b[D`` (left) at the start of the value. The receiving
        server rejects the resulting Authorization header with HTTP 400.
        """
        polluted = "\x1b[D\x1b[Ask-or-v1-abc123"
        result = _check_non_ascii_credential("OPENROUTER_API_KEY", polluted)
        assert result == "[D[Ask-or-v1-abc123"
        assert "\x1b" not in result
        captured = capsys.readouterr()
        assert "U+001B" in captured.err

    def test_strips_ascii_control_bytes(self, capsys):
        """Any ASCII control byte (0x00-0x1F, 0x7F) is invalid in headers."""
        result = _check_non_ascii_credential("TEST_API_KEY", "sk-\x00\x07\x7fproj-abc")
        assert result == "sk-proj-abc"
        captured = capsys.readouterr()
        assert "break API requests" in captured.err

    def test_empty_key(self):
        result = _check_non_ascii_credential("TEST_KEY", "")
        assert result == ""

    def test_all_ascii_no_warning(self, capsys):
        result = _check_non_ascii_credential("KEY", "all-ascii-value-123")
        assert result == "all-ascii-value-123"
        captured = capsys.readouterr()
        assert captured.err == ""


class TestEnvLoaderSanitization:
    """Tests for _sanitize_loaded_credentials in env_loader."""

    def test_strips_non_ascii_from_api_key(self, monkeypatch):
        from hermes_cli.env_loader import _sanitize_loaded_credentials, _WARNED_KEYS

        _WARNED_KEYS.discard("OPENROUTER_API_KEY")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-proj-abcʋdef")
        _sanitize_loaded_credentials()
        assert os.environ["OPENROUTER_API_KEY"] == "sk-proj-abcdef"

    def test_strips_non_ascii_from_token(self, monkeypatch):
        from hermes_cli.env_loader import _sanitize_loaded_credentials, _WARNED_KEYS

        _WARNED_KEYS.discard("DISCORD_BOT_TOKEN")
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tokénvalue")
        _sanitize_loaded_credentials()
        assert os.environ["DISCORD_BOT_TOKEN"] == "toknvalue"

    def test_ignores_non_credential_vars(self, monkeypatch):
        from hermes_cli.env_loader import _sanitize_loaded_credentials

        monkeypatch.setenv("MY_UNICODE_VAR", "héllo wörld")
        _sanitize_loaded_credentials()
        # Not a credential suffix — should be left alone
        assert os.environ["MY_UNICODE_VAR"] == "héllo wörld"

    def test_ascii_credentials_untouched(self, monkeypatch):
        from hermes_cli.env_loader import _sanitize_loaded_credentials

        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-allascii123")
        _sanitize_loaded_credentials()
        assert os.environ["OPENAI_API_KEY"] == "sk-proj-allascii123"

    def test_warns_to_stderr_when_stripping(self, monkeypatch, capsys):
        """Silent stripping masks bad keys as opaque provider 400s (see #6843 fallout).

        Users must be told when a copy-paste artifact was removed so they
        can re-copy the key if authentication fails.
        """
        from hermes_cli.env_loader import _sanitize_loaded_credentials, _WARNED_KEYS

        _WARNED_KEYS.discard("GOOGLE_API_KEY")
        monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSy\u200babcdef")  # ZWSP mid-key
        _sanitize_loaded_credentials()
        assert os.environ["GOOGLE_API_KEY"] == "AIzaSyabcdef"

        captured = capsys.readouterr()
        assert "GOOGLE_API_KEY" in captured.err
        assert "U+200B" in captured.err
        assert "re-copy" in captured.err.lower()

    def test_warning_fires_only_once_per_key(self, monkeypatch, capsys):
        """Repeated loads (user env + project env) must not double-warn."""
        from hermes_cli.env_loader import _sanitize_loaded_credentials, _WARNED_KEYS

        _WARNED_KEYS.discard("GEMINI_API_KEY")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza\u028bbad")
        _sanitize_loaded_credentials()
        first = capsys.readouterr().err

        monkeypatch.setenv("GEMINI_API_KEY", "AIza\u028bbad2")
        _sanitize_loaded_credentials()
        second = capsys.readouterr().err

        assert "GEMINI_API_KEY" in first
        assert second == ""  # no repeat warning

    def test_strips_ascii_control_chars(self, monkeypatch, capsys):
        """ASCII control bytes (e.g. ESC 0x1B from terminal paste) are invalid
        in HTTP header values per RFC 7230 and get stripped.

        Real-world repro: a user typed an API key into the setup prompt
        and pressed arrow keys during input, which captured ESC control
        sequences into the saved value. The receiving server rejected
        the resulting Authorization header with HTTP 400 before parsing
        the body.
        """
        from hermes_cli.env_loader import _sanitize_loaded_credentials, _WARNED_KEYS

        _WARNED_KEYS.discard("ANTHROPIC_API_KEY")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant\x1bapi-key")
        _sanitize_loaded_credentials()
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-antapi-key"
        captured = capsys.readouterr()
        assert "ANTHROPIC_API_KEY" in captured.err
        assert "U+001B" in captured.err

    def test_strips_ansi_arrow_key_prefix(self, monkeypatch, capsys):
        """The exact pollution pattern from the field: ``\\x1b[D\\x1b[A`` glued
        to the front of an otherwise valid key by arrow-key terminal input."""
        from hermes_cli.env_loader import _sanitize_loaded_credentials, _WARNED_KEYS

        _WARNED_KEYS.discard("OPENROUTER_API_KEY")
        monkeypatch.setenv("OPENROUTER_API_KEY", "\x1b[D\x1b[Ask-or-v1-abc123")
        _sanitize_loaded_credentials()
        # ESC bytes stripped; the remaining "[D" / "[A" prefix is harmless
        # ASCII (still rejected by the provider as an invalid key, but at
        # least the HTTP request reaches the provider with a parseable
        # Authorization header so the error message is actionable).
        assert os.environ["OPENROUTER_API_KEY"] == "[D[Ask-or-v1-abc123"
        captured = capsys.readouterr()
        assert "U+001B" in captured.err
