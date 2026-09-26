# Security

## Reporting a vulnerability

Please report security issues privately through GitHub's "Report a vulnerability" button on the
repository's Security tab. Do not open a public issue. You will get a reply within a week.

Include what you found, how to reproduce it, and what an attacker could do with it.

## What skill-issue does and does not do

skill-issue reads skill files and decides which ones to mention to your coding agent. Its threat
model follows from that.

**It never executes third-party skill code.** Indexing, routing, and `skill-issue scan` only read
files. Skill scripts are listed, hashed and scanned as text.

**It never installs a skill on its own.** In approved mode the router can suggest a skill that
is not installed. The injected context tells the agent to ask you first. `skill-issue install`
only runs when you run it, and it requires either a typed `yes` on a terminal or
`--confirm <first 12 chars of the sha256>`, so an agent cannot install something without the
hash it was shown.

**Installs are hash-pinned.** Every approved skill is pinned to a repository, a commit, and a
SHA-256 over all of its files. The installer downloads that exact commit, extracts only the skill
folder (no symlinks, no path traversal), recomputes the hash, and refuses on any mismatch.

**The daemon listens on 127.0.0.1 only** and every request must carry a random per-start token
from `~/.skill-issue/run/daemon.json` (file mode 0600 on POSIX). Browsers cannot attach that
header cross-origin without a CORS preflight, which the daemon does not answer.

**Nothing leaves your machine** during routing. Network access happens only when you download
model weights (the fine-tuned gates from this repository's GitHub release, the embedder and base
models from Hugging Face), fetch the approved index, or install an approved skill. Release
weights are checked against a SHA-256 pinned in `src/skillissue/models.py` before they are
unpacked, and the unpacker refuses paths that escape the target folder. The
optional TypeSafe gate sends requests to TypeSafe's API, and only if you set `TYPESAFE_API_KEY`
and choose that gate.

## Limits of the static scanner

`skill-issue scan` and the approved-index build use regex rules for network calls,
credential and environment access, shell and dynamic code execution, obfuscation, and
instruction-like text aimed at the agent. This catches the obvious and the lazy. It does not
prove a skill is safe: a determined author can write code that no regex flags. Treat the scan as
one signal next to the source repository, the trust tier, and your own reading of the files.

## Prompt injection

Skill descriptions are third-party text and end up in your agent's context, both through
Claude Code's own skill listing and through skill-issue's injected block. skill-issue only
injects names and descriptions of skills that passed its gate, wraps them in a
`<skill-issue>` block, and never injects skill bodies. It does not make a malicious installed
skill safe. Install skills you trust.
