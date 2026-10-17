from rag_anonymous_metrics import environment


def test_describe_environment_reports_every_field():
    info = environment.describe_environment()

    assert set(info) == {
        "os",
        "python",
        "cpu",
        "cpu_cores",
        "ram_gb",
        "gpu",
        "ollama",
        "rag_anonymous_metrics",
        "rag_anonymous",
    }
    assert all(isinstance(v, str) and v for v in info.values())


def test_missing_tools_become_unknown(monkeypatch):
    monkeypatch.setattr(environment, "_run", lambda *command: None)
    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    monkeypatch.setattr(environment.platform, "processor", lambda: "")
    monkeypatch.setattr(environment.urllib.request, "urlopen", _refuse)
    monkeypatch.setattr(environment, "open", _refuse, raising=False)

    info = environment.describe_environment()

    assert info["gpu"] == environment.UNKNOWN
    assert info["ollama"] == environment.UNKNOWN
    assert info["cpu"] == environment.UNKNOWN


def _refuse(*args, **kwargs):
    raise OSError("unavailable")
