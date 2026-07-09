from unittest.mock import Mock, patch

from src.test_frontend import discover_test_files, render_index, run_pytest, safe_test_targets


def test_discover_test_files_returns_relative_sorted_paths():
    files = discover_test_files()

    assert files == sorted(files)
    assert "tests/test_wiki_navigator.py" in files
    assert all(path.startswith("tests/test_") and path.endswith(".py") for path in files)


def test_safe_test_targets_rejects_unknown_and_path_traversal():
    selected = safe_test_targets(["tests/test_wiki_navigator.py", "../secrets.py", "not-a-test.py"])

    assert selected == ["tests/test_wiki_navigator.py"]


def test_safe_test_targets_defaults_to_full_suite_when_empty_or_invalid():
    assert safe_test_targets([]) == ["tests"]
    assert safe_test_targets(["../bad.py"]) == ["tests"]


def test_render_index_includes_test_checkboxes_and_api_endpoint():
    html = render_index(["tests/test_example.py"])

    assert "Wikipedia SpeedRun Test Runner" in html
    assert "tests/test_example.py" in html
    assert "/api/run-tests" in html


def test_run_pytest_captures_command_and_result():
    completed = Mock(returncode=0, stdout="ok", stderr="")
    with patch("src.test_frontend.subprocess.run", return_value=completed) as subprocess_run:
        result = run_pytest(["tests/test_wiki_navigator.py"], extra_args=["-q"])

    assert result.passed is True
    assert result.exit_code == 0
    assert result.stdout == "ok"
    assert result.command[-2:] == ["tests/test_wiki_navigator.py", "-q"]
    subprocess_run.assert_called_once()
    assert subprocess_run.call_args.kwargs["check"] is False
