# HaoLine Cursor Agent Rules

> Rules for AI agents (Cursor, OpenHands, ChatGPT) operating on this repository.
> These rules supplement the project's `.cursorrules` file.

---

## 1. Before Starting Work

### 1.1. Read the Memory Bank

Before doing any significant work, read these files:

1. `memory_bank/INDEX.md` - Quick reference to all docs
2. `memory_bank/GLOSSARY.md` - Understand project terminology
3. `memory_bank/DECISIONS.md` - Recent architectural decisions
4. `BACKLOG.md` - Current work items and priorities
5. `PRD.md` - Product requirements (especially Delta Log section)

### 1.2. Understand the Codebase

- Core analysis: `src/haoline/analyzer.py`, `src/haoline/report.py`
- Universal IR: `src/haoline/universal_ir.py`
- Format readers: `src/haoline/formats/`
- CLI: `src/haoline/cli.py`
- Web UI: `src/haoline/streamlit_app.py`

### 1.3. Check Current State

```bash
git status                    # Any uncommitted changes?
git log --oneline -5          # Recent commits
```

---

## 2. Code Style (ALWAYS Follow)

### 2.1. Python Formatting

- **Ruff-formatted**: 100 char line limit, double quotes, trailing commas
- **Ruff-linted**: No unused imports, proper import sorting
- **Type hints**: All function signatures (args and return types)
- **Docstrings**: Google-style for public functions/classes

### 2.2. Before Every Commit

```bash
ruff format src/haoline/
ruff check src/haoline/ --fix
```

### 2.3. Every 3 Commits (or Before Release)

```bash
mypy src/haoline/ --ignore-missing-imports
```

### 2.4. Commit Messages

Use conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

---

## 3. Agent Mode Cadence

Use exactly one MODE per reply: **ASK**, **PLAN**, **BUILD**, **TEST**, or **DOCS**.

**Rule:** At least once every 5 replies, choose a non-BUILD mode (ASK/PLAN/TEST/DOCS).

### 3.1. Reply Footer (Always Include)

```
Mode: <ASK|PLAN|BUILD|TEST|DOCS>
Evidence: <commands run / outputs / artifact paths>
Checkpoint: <N turns until next required non-BUILD>
```

---

## 4. Memory Bank Updates

### 4.1. When to Update DECISIONS.md

After you:
- Make a non-obvious architectural choice
- Choose between multiple implementation approaches
- Encounter and resolve a significant blocker
- Change a public API or data schema

### 4.2. When to Update BACKLOG.md

After you:
- Complete a task, story, or epic
- Discover a new blocker
- Add or remove work items
- Change priorities

### 4.3. When to Update GLOSSARY.md

When you:
- Introduce new terminology
- Create new abstractions
- Notice undocumented concepts being used

### 4.4. When to Update PRD.md (Delta Log)

After any significant session, add a brief entry to the Delta Log:
- Date
- What was accomplished
- Any blockers or decisions
- Next steps

---

## 5. Dependency Management

When adding a new optional dependency:

1. Add to `pyproject.toml` under `[project.optional-dependencies]`
2. Create dedicated extra if format/feature-specific
3. Add to `full` extra if cross-platform
4. **Update README.md** extras table
5. If format reader, update `test_format_readers.py`

---

## 6. Version Compatibility (HuggingFace Spaces)

`streamlit_app.py` runs against PyPI version, not latest code. When accessing NEW attributes on core classes:

```python
# BAD - crashes if attribute doesn't exist in PyPI version:
if report.universal_graph:
    ...

# GOOD - safe access with fallback:
if hasattr(report, "universal_graph") and report.universal_graph:
    ...
```

---

## 7. Testing

- Run `pytest src/haoline/tests/ -v` before major changes
- Use `pytest.mark.skipif` for optional dependencies
- Format reader tests go in `test_format_readers.py` at repo root

---

## 8. Anti-Patterns to Avoid

### 8.1. Monochat Syndrome

Don't assume context from previous sessions. Read the memory bank.

### 8.2. Over-Engineering

Only make changes directly requested. Don't add features, refactor code, or make "improvements" beyond what was asked.

### 8.3. Duplicate Files

Before creating a new file, list same-type files already present in that directory.

### 8.4. Secret Knowledge

If you learn something important, write it to the memory bank. If it's not in Git, it didn't happen.

---

## 9. Quick Reference

| Task | Command/Location |
|------|------------------|
| Format code | `ruff format src/haoline/` |
| Lint code | `ruff check src/haoline/ --fix` |
| Type check | `mypy src/haoline/ --ignore-missing-imports` |
| Run tests | `pytest src/haoline/tests/ -v` |
| Current backlog | `BACKLOG.md` |
| Recent decisions | `memory_bank/DECISIONS.md` |
| Project terms | `memory_bank/GLOSSARY.md` |

---

*These rules are designed to make AI agents effective collaborators on long-lived projects.*

