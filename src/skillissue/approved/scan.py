"""Static scanner for third-party skills. Reads files, never runs them.

Flags five things: network calls, credential or env access, shell/code execution, obfuscation,
and instruction-like text aimed at the agent (prompt injection). Each finding has a severity;
any `high` finding fails the scan.

Severity depends on where a match sits. In a script the code runs, so `curl ... | sh` is high.
In SKILL.md or a reference doc the same text is usually a normal install instruction (official
installers look exactly like that), so it is medium there.

This is a tripwire, not a guarantee. It catches the obvious and the lazy. A determined author
can evade regexes, which is why approved-mode installs also pin a content hash and always need
explicit user confirmation."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..skill import MAX_TEXT_BYTES, SCRIPT_SUFFIXES


@dataclass
class Finding:
    rule: str
    category: str
    severity: str  # medium | high
    file: str
    line: int
    excerpt: str


@dataclass
class Rule:
    name: str
    category: str
    # Severity per file scope ("script" or "doc"); a scope missing here is not checked.
    severity: dict[str, str]
    pattern: re.Pattern[str]


def _r(name: str, category: str, script: str | None, doc: str | None, pattern: str, flags: int = re.I) -> Rule:
    sev = {k: v for k, v in (("script", script), ("doc", doc)) if v}
    return Rule(name, category, sev, re.compile(pattern, flags))


RULES: list[Rule] = [
    # --- remote code execution / download-and-run ---
    _r("pipe-to-shell", "exec", "high", "medium",
       r"(curl|wget|iwr|invoke-webrequest)\b[^\n|]*\|\s*(sudo\s+)?(ba|z|da)?sh\b|\|\s*iex\b"),
    _r("download-exec", "exec", "high", "medium",
       r"(curl|wget)[^\n]*(-o|>)\s*\S+[^\n]*&&\s*(chmod\s+\+x|\./|sh\s|bash\s|python\s)"),
    # --- network calls in scripts ---
    _r("net-python", "network", "medium", None,
       r"\b(requests\.(get|post|put|delete|request)|urllib\.request|http\.client|httpx\.|aiohttp|socket\.socket|urlopen)\b"),
    _r("net-js", "network", "medium", None,
       r"\b(fetch\s*\(|axios\.|XMLHttpRequest|https?\.request|net\.connect|WebSocket\()"),
    _r("net-shell", "network", "medium", None,
       r"(^|[\s;|&(])(curl|wget|nc|ncat|scp|rsync|Invoke-WebRequest|Invoke-RestMethod)\s"),
    # --- credentials / environment ---
    _r("secret-files", "credentials", "high", "medium",
       r"[/\\]\.ssh[/\\](id_(rsa|dsa|ecdsa|ed25519)(?!\.pub)\b|[a-z_]+_key\b)|\.aws[/\\]credentials|\.git-credentials"
       r"|[/\\]\.netrc\b|\.docker[/\\]config\.json|\bwallet\.dat\b|Login Data|Cookies\.sqlite"
       r"|security\s+find-(generic|internet)-password"),
    _r("env-secrets", "credentials", "medium", None,
       r"(os\.environ|os\.getenv|process\.env|\$env:|getenv)\W{0,3}[\[(.]?\W{0,2}[A-Z_]*(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)", 0),
    _r("env-dump", "credentials", "high", "high",
       r"\bprintenv\s*(\||>(?!\s*/dev/null))|\benv\s*\|\s*(curl|nc|base64)|JSON\.stringify\(process\.env\)"
       r"|dict\(os\.environ\)[^\n]*(post|send|write)"),
    # --- shell / dynamic code execution ---
    _r("py-exec", "exec", "medium", None,
       r"\b(subprocess\.(run|call|Popen|check_output|check_call)|os\.system|os\.popen|pty\.spawn)\b"),
    _r("py-shell-true", "exec", "medium", None, r"shell\s*=\s*True"),
    _r("dyn-eval-encoded", "exec", "high", None,
       r"(?<![\w.])(eval|exec)\s*\([^)\n]{0,40}(b64decode|base64|codecs\.decode|zlib\.decompress|atob|bytes\.fromhex|requests\.|urlopen)"),
    _r("dyn-eval", "exec", "medium", None, r"(?<![\w.])(eval|exec)\s*\((?!\s*\))"),
    _r("js-exec", "exec", "medium", None, r"\b(child_process|execSync|spawnSync|new Function\()"),
    _r("destructive", "exec", "high", "high",
       r"\brm\s+-rf\s+(/|~|\$HOME)(\s|$|\*)|\bmkfs\.\w+\s+/dev/|\bdd\s+if=\S+\s+of=/dev/(sd|nvme|disk)|:\(\)\s*\{\s*:\|:&\s*\};:"),
    _r("persistence", "exec", "high", "medium",
       r"crontab\s+-[lre]?\s*[<|]|/etc/cron\.|(>>?|tee\s+-a)\s*(~|\$HOME)/\.(bashrc|zshrc|profile|bash_profile)"
       r"|Library/LaunchAgents/|CurrentVersion\\+Run\b"),
    # --- obfuscation ---
    _r("base64-blob", "obfuscation", "high", "medium", r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{300,}={0,2}(?![A-Za-z0-9+/])", 0),
    _r("b64-decode-run", "obfuscation", "high", "high",
       r"base64\s+(-d|--decode)[^\n]*\|\s*(ba)?sh|b64decode\([^)]*\)[^\n]{0,40}(exec|eval|system)\("),
    _r("hex-escapes", "obfuscation", "medium", "medium", r"(\\x[0-9a-fA-F]{2}){32,}", 0),
    _r("bidi-controls", "obfuscation", "high", "high", "[\u202a-\u202e\u2066-\u2069]", 0),
    _r("zero-width-run", "obfuscation", "medium", "medium", "[\u200b\u200c\u2060]{3,}", 0),
    # --- prompt injection aimed at the agent ---
    _r("ignore-instructions", "injection", None, "high",
       r"(?<![\"'\u201c\u2018])\b(ignore|disregard|forget)\s+(all\s+|any\s+)?(of\s+)?(your|the|my)?\s*"
       r"(previous|prior|above|earlier|system|safety|original)\s+(instructions|rules|prompts?|guidelines|guardrails)"),
    _r("override-instructions", "injection", None, "medium",
       r"\b(override|supersede|take precedence over)s?\s+(all|any)\s+(other\s+)?(instructions|rules|system prompt)"),
    _r("hide-from-user", "injection", None, "high",
       r"\b(do not|don't|never)\s+(tell|inform|reveal|disclose|mention|show)\s+(this|these|that|it|anything)?\s*(to\s+)?"
       r"the user\b(?!'s|\s+(to|how|about how|whether|which|what|if)\b)|\bwithout (telling|informing|notifying) the user\b"
       r"|\b(hide|conceal|keep)\s+(this|it|these)\s+(secret\s+)?from the user\b"),
    _r("secret-exfil", "injection", None, "high",
       r"\b(send|upload|post|forward|exfiltrate|transmit)\b[^\n.]{0,60}\b(api[_ -]?keys?|credentials|tokens?|secrets?|passwords?|ssh keys?|\.env)\b"
       r"[^\n.]{0,40}\b(to|at)\s+(https?://|the (server|endpoint|webhook))"),
    _r("role-hijack", "injection", None, "medium",
       r"\byou are (now|no longer) (a|an|in|bound)\b|\bnew system prompt\b|\bact as (an? )?(unrestricted|jailbroken)|\bdeveloper mode enabled\b"),
    _r("auto-approve", "injection", None, "medium",
       r"--dangerously-skip-permissions|\bbypassPermissions\b|\bauto[- ]?approve (all|every)\b|\bdisable (the )?(sandbox|permission prompts)"),
    _r("hidden-comment-instruction", "injection", None, "medium",
       r"<!--(?:(?!-->).){0,400}\b(assistant|claude|agent|AI|model)\b(?:(?!-->).){0,400}\b(must|should|always|never|ignore)\b(?:(?!-->).){0,400}-->",
       re.I | re.S),
]


def _scope_of(path: Path) -> str:
    """Scripts run; everything else (SKILL.md, references, configs) is read by the agent."""
    return "script" if path.suffix.lower() in SCRIPT_SUFFIXES else "doc"


def scan_text(rel: str, text: str, scope: str) -> list[Finding]:
    out: list[Finding] = []
    for rule in RULES:
        sev = rule.severity.get(scope)
        if sev is None:
            continue
        n = 0
        for m in rule.pattern.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            excerpt = text[max(0, m.start() - 20): m.end() + 20].replace("\n", " ")
            out.append(Finding(rule.name, rule.category, sev, rel, line, excerpt[:160]))
            n += 1
            if n >= 3:
                break  # enough examples of this rule in this file
    return out


def scan_files(files: Iterable[tuple[str, str]]) -> list[Finding]:
    """files: (relative path, text)."""
    findings: list[Finding] = []
    for rel, text in files:
        findings += scan_text(rel, text, _scope_of(Path(rel)))
    return findings


def scan_dir(skill_dir: Path) -> list[Finding]:
    items = []
    for p in sorted(skill_dir.rglob("*")):
        if not p.is_file() or p.stat().st_size > MAX_TEXT_BYTES:
            continue
        try:
            text = p.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue  # binary
        items.append((p.relative_to(skill_dir).as_posix(), text))
    return scan_files(items)


def verdict(findings: list[Finding]) -> str:
    if any(f.severity == "high" for f in findings):
        return "fail"
    if findings:
        return "warn"
    return "pass"


def summarize(findings: list[Finding]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.rule] = counts.get(f.rule, 0) + 1
    return {
        "verdict": verdict(findings),
        "counts": counts,
        "high": [asdict(f) for f in findings if f.severity == "high"][:10],
    }
