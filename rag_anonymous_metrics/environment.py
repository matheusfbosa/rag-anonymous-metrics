import json
import logging
import os
import platform
import subprocess
import urllib.request
from importlib import metadata
from pathlib import Path

import rag_anonymous
from rag_anonymous import Settings as AnonSettings
from rag_anonymous_metrics.config import PROJECT_ROOT

UNKNOWN = "unknown"


def describe_environment() -> dict[str, str]:
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "cpu_cores": str(os.cpu_count() or UNKNOWN),
        "ram_gb": _ram_gb(),
        "gpu": _gpu(),
        "ollama": _ollama_version(),
        "rag_anonymous_metrics": _code_version(PROJECT_ROOT, "rag-anonymous-metrics"),
        "rag_anonymous": _code_version(
            Path(rag_anonymous.__file__).resolve().parent.parent, "rag-anonymous"
        ),
    }


def log_environment(logger: logging.Logger) -> None:
    for key, value in describe_environment().items():
        logger.info("Environment: %s=%s", key, value)


def _run(*command: str) -> str | None:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _cpu_model() -> str:
    if platform.system() == "Darwin":
        return _run("sysctl", "-n", "machdep.cpu.brand_string") or UNKNOWN
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or UNKNOWN


def _ram_gb() -> str:
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return UNKNOWN
    return f"{total / 1024**3:.1f}"


def _gpu() -> str:
    nvidia = _run(
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader",
    )
    if nvidia:
        return "; ".join(line.strip() for line in nvidia.splitlines())
    if platform.system() == "Darwin":
        displays = _run("system_profiler", "SPDisplaysDataType") or ""
        for line in displays.splitlines():
            if "Chipset Model" in line:
                return line.split(":", 1)[1].strip() + " (unified memory)"
    return UNKNOWN


def _ollama_version() -> str:
    url = f"{AnonSettings.load().ollama_base_url.rstrip('/')}/api/version"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return str(json.load(response).get("version", UNKNOWN))
    except (OSError, ValueError):
        return UNKNOWN


def _code_version(repo: Path, package: str) -> str:
    commit = _run("git", "-C", str(repo), "rev-parse", "--short", "HEAD")
    if commit:
        dirty = _run("git", "-C", str(repo), "status", "--porcelain")
        return f"git {commit}{' (uncommitted changes)' if dirty else ''}"
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return UNKNOWN
