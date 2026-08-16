"""The direct-model benchmark must be reproducible.

`ollama run` accepts no sampling parameters, so calling it inherits the temperature
set in the Modelfile. A benchmark built on that is not measuring the model: the same
model over the same cases produces a different score on every run, and that spread
can be larger than the gap between the two models you are comparing.

These tests pin the contract: temperature 0 and a fixed seed on every call.
"""

import json
from unittest.mock import MagicMock, patch

from run_eval_benchmark import DETERMINISTIC_OPTIONS, query_ollama


def _fake_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps({"response": text}).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = lambda *a: None
    return resp


def test_sampling_is_deterministic_by_default():
    """Without this, benchmark runs cannot be compared to each other."""
    assert DETERMINISTIC_OPTIONS["temperature"] == 0.0
    assert DETERMINISTIC_OPTIONS["seed"] is not None
    assert DETERMINISTIC_OPTIONS["top_k"] == 1


def test_query_sends_deterministic_options():
    """Declaring the options is not enough — they must reach the request body."""
    with patch("urllib.request.urlopen", return_value=_fake_response("ok")) as mock_open:
        text, latency = query_ollama("qwen2.5-coder:7b", "hello")

    sent = json.loads(mock_open.call_args[0][0].data)
    assert sent["options"]["temperature"] == 0.0
    assert sent["options"]["seed"] == DETERMINISTIC_OPTIONS["seed"]
    assert sent["stream"] is False
    assert text == "ok"
    assert latency >= 0


def test_caller_can_override_without_losing_the_rest():
    """An explicit override should not silently drop the other guarantees."""
    with patch("urllib.request.urlopen", return_value=_fake_response("x")) as mock_open:
        query_ollama("qwen2.5-coder:7b", "p", options={"temperature": 0.7})

    sent = json.loads(mock_open.call_args[0][0].data)
    assert sent["options"]["temperature"] == 0.7
    assert sent["options"]["seed"] == DETERMINISTIC_OPTIONS["seed"]


def test_errors_return_a_marker_instead_of_raising():
    """One failed call must not abort a run of many cases."""
    with patch("urllib.request.urlopen", side_effect=OSError("boom")):
        text, _ = query_ollama("qwen2.5-coder:7b", "p")
    assert text.startswith("[ERROR")
