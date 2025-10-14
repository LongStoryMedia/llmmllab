import os
import json
import pytest


def _skip(msg: str):  # helper to centralize skip logic
    pytest.skip(msg)


@pytest.mark.timeout(60)
def test_llama_cpp_stream_includes_logprobs_when_requested():
    """Integration-style test: verify that passing logprobs/top_logprobs per request
    yields streaming chunks containing logprob data.

    This targets a running llama.cpp compatible server (like the one shown in the
    user curl example) exposing the OpenAI-compatible /v1/chat/completions endpoint.

    Requirements:
      - Environment variable LLAMA_CPP_SERVER_URL set, e.g. http://localhost:3000
      - A model already loaded/serving on that server.

    If the server/env isn't present, the test is skipped (so it won't fail CI).
    """
    server_url = os.getenv("LLAMA_CPP_SERVER_URL")
    if not server_url:
        _skip("LLAMA_CPP_SERVER_URL not set; skipping live llama.cpp logprobs test")

    try:
        import requests  # type: ignore
    except ImportError:  # pragma: no cover
        _skip("requests not installed; skipping live llama.cpp logprobs test")

    endpoint = server_url.rstrip("/") + "/v1/chat/completions"

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Briefly name the largest animal."},
        ],
        "stream": True,
        "temperature": 0.2,
        "max_tokens": 32,
        # Critical: request-side parameters we want to validate
        "logprobs": True,
        "top_logprobs": 5,
    }

    has_logprobs = False
    first_chunk_seen = False

    with requests.post(endpoint, json=payload, stream=True, timeout=45) as resp:
        if resp.status_code != 200:
            _skip(f"llama.cpp server returned status {resp.status_code}; skipping")

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if not raw_line.startswith("data: "):
                continue
            line = raw_line[len("data: ") :].strip()
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                # Ignore malformed lines (some servers may emit keepalives)
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            first_chunk_seen = True
            choice0 = choices[0]
            # In the sample output, logprobs nested under delta.logprobs
            delta = choice0.get("delta", {}) or {}
            logprobs_block = delta.get("logprobs") or choice0.get("logprobs")
            if logprobs_block and isinstance(logprobs_block, dict):
                content_entries = logprobs_block.get("content")
                if content_entries and isinstance(content_entries, list):
                    # Validate expected keys for at least one token
                    token_entry = content_entries[0]
                    assert "token" in token_entry
                    assert "logprob" in token_entry
                    assert "top_logprobs" in token_entry
                    has_logprobs = True
                    break

    assert (
        first_chunk_seen
    ), "Did not receive any streaming chunks from llama.cpp server"
    assert (
        has_logprobs
    ), "Streaming chunks lacked logprobs despite requesting them per call"


@pytest.mark.timeout(60)
def test_llama_cpp_stream_omits_logprobs_when_not_requested():
    """Complementary test: requesting streaming WITHOUT logprobs should yield
    chunks without logprob metadata. Skipped if server unavailable.
    """
    server_url = os.getenv("LLAMA_CPP_SERVER_URL")
    if not server_url:
        _skip("LLAMA_CPP_SERVER_URL not set; skipping live llama.cpp omission test")

    try:
        import requests  # type: ignore
    except ImportError:  # pragma: no cover
        _skip("requests not installed; skipping live llama.cpp omission test")

    endpoint = server_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say a single color."},
        ],
        "stream": True,
        "temperature": 0.2,
        "max_tokens": 8,
        # Intentionally omit logprobs/top_logprobs
    }

    saw_chunk = False
    saw_any_logprobs = False

    import requests  # safe (already guarded)

    with requests.post(endpoint, json=payload, stream=True, timeout=45) as resp:
        if resp.status_code != 200:
            _skip(f"llama.cpp server returned status {resp.status_code}; skipping")
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if not raw_line.startswith("data: "):
                continue
            line = raw_line[len("data: ") :].strip()
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            saw_chunk = True
            choice0 = choices[0]
            delta = choice0.get("delta", {}) or {}
            if delta.get("logprobs") or choice0.get("logprobs"):
                saw_any_logprobs = True
                break

    assert saw_chunk, "Did not receive any streaming chunks for omission test"
    assert not saw_any_logprobs, "Unexpected logprobs present when not requested"
