---
name: skill-issue
description: Find which installed Agent Skills fit the current task by asking the local skill-issue router, and handle its suggestions safely. Use at the start of any non-trivial task when no <skill-issue> context block is already present, and whenever the user asks why a skill did or did not trigger, wants to switch skill-issue modes, or wants to install a skill the router suggested.
---

# skill-issue: consult the skill router

skill-issue is a local router. Given a request, it returns the few skills worth loading, or none.
Returning none is normal and common. It runs on this machine; nothing is sent anywhere.

## When a `<skill-issue>` block is already in context

The Claude Code plugin already routed this prompt. Do not call the router again. If a listed
skill fits, load it before starting. Ignore listed skills that do not fit.

## How to consult the router

Prefer the MCP tool if it is available:

- `find_skills(request, cwd)` returns `{"skills": [...]}`. Each installed skill has a
  `skill_md` path. Read that file and follow it.
- `explain_route(request, cwd)` shows every candidate with its probability and retrieval rank.
  Use it when the user asks why something was or was not picked.

Otherwise use the CLI:

```bash
skill-issue route "<the user's request, verbatim>" --json
```

Read `selected`. For each selected skill with `"installed": true`, read `<path>/SKILL.md` and
follow it. If `selected` is empty, continue without a skill. Do not keep retrying with reworded
requests to force a match.

## Skills that are not installed (approved mode)

In approved mode the router can suggest a vetted public skill marked `"installed": false`.

1. Never install it on your own.
2. Show the user its name, source repo and commit, trust tier, license, sha256, and the list of
   scripts it contains. Say that installing copies those files into their skills folder.
3. Only if the user explicitly says yes, run:
   `skill-issue install <name> --confirm <first 12 characters of the sha256>`
4. The command re-verifies the hash and refuses on any mismatch. Report what it printed.

## Other commands the user may ask for

| Request | Command |
|---|---|
| Switch modes | `skill-issue mode installed` or `skill-issue mode approved` |
| Check the setup | `skill-issue doctor` |
| See what the router can see | `skill-issue index` |
| Scan a skill folder for risky content | `skill-issue scan <path>` |
| Browse the approved index | `skill-issue approved <search words>` |

`skill-issue scan` only reads files. Never run a third-party skill's scripts to "test" it.
