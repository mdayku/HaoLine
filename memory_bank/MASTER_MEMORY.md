# HaoLine Memory Bank & Monorepo Playbook

> **Purpose:** This document defines how HaoLine uses the repo itself as a shared "memory bank" for humans and AI agents, so we don't rely on a single chat history ("monochat") to remember anything important.

---

## 0. TL;DR

- **The repo is the source of truth.** Not any one ChatGPT/Cursor session.
- We maintain a small set of **living docs** (PRDs, backlog, decisions, glossary, BrainLift) that act as the **memory bank**.
- **AI agents are explicitly instructed** to:
  - Read these docs before doing serious work.
  - Use some non-build turns to keep them updated.
- **Humans treat memory docs like code**: review them in PRs, keep them current, and avoid side-channel "secret knowledge."

---

## 1. Goals

1. **Durable memory across chats and tools**
   - Decisions, conventions, and context live in Git, not in someone's browser tab.
2. **Make AI useful over long-lived projects**
   - Agents can come in "cold", read the memory bank, and behave like they've been on the project for weeks.
3. **Collaboration-friendly**
   - Everyone (humans + agents) share the same ground truth and update it through normal commits/PRs.

---

## 2. Directory & File Structure

Layout inside the HaoLine repo:

```text
memory_bank/
  MASTER_MEMORY.md       # This file: explains the system + rules for agents & humans
  INDEX.md               # Lightweight table of contents / pointers to key docs
  DECISIONS.md           # Chronological log of important decisions and tradeoffs
  GLOSSARY.md            # Project-specific terms, concepts, shorthand
  CURSOR_AGENT_RULES.md  # Instructions specifically for Cursor / OpenHands agents
```

Elsewhere in the repo:

```text
PRD.md                   # Product Requirements for HaoLine
BACKLOG.md               # High-level backlog / epics (42+ epics tracked)
Architecture.md          # System architecture and design
DEPLOYMENT.md            # Deployment guide for PyPI and HuggingFace Spaces
PRIVACY.md               # Privacy guarantees and offline mode
README.md                # User-facing documentation
.cursorrules             # Cursor-specific coding conventions
```

External (not in this repo):

```text
C:\Users\marcu\onnxruntime\docs\marcu\BRAINLIFT.md  # BrainLift-style learning log
```

---

## 3. Anti-"Monochat" Philosophy

**Monochat** = the bad pattern where each new chat/agent session starts from zero context and reinvents the same decisions.

This repo is the opposite:

* The **memory bank is the "long-term memory"**.
* Chat sessions are **stateless workers** that:
  * Read from the memory bank,
  * Do focused work,
  * Write back anything important.

If it's not in Git, it basically didn't happen.

---

## 4. What Belongs in the Memory Bank?

Think of this as **things future you (or a new teammate) would wish were written down**.

### 4.1. MASTER_MEMORY.md (this file)

* Purpose and philosophy (this doc).
* Rules for how we use the memory bank.
* Pointers to the other docs.

### 4.2. INDEX.md

* Simple table of contents linking to all key docs.
* Update when new long-lived docs are added or files are moved.

### 4.3. DECISIONS.md

* A chronological log of "we decided X, not Y, because Z".
* Small, bullet-point entries with date, decision, alternatives, and reasoning.

### 4.4. GLOSSARY.md

* Short definitions of HaoLine-specific language and concepts.

### 4.5. CURSOR_AGENT_RULES.md

* Concrete, **promptable rules** for Cursor / OpenHands agents.
* Synced with `.cursorrules` content.

---

## 5. Rules for AI Agents (Cursor / OpenHands / ChatGPT)

**If you are an AI agent operating on this repo, follow these rules.**

### 5.1. Before Doing Real Work

1. **Always read the memory index:**
   * Open `memory_bank/INDEX.md`.
   * Skim linked docs that are relevant to the task (PRD, BACKLOG, DECISIONS, etc.).

2. **Respect the PRD and backlog:**
   * Don't propose features that contradict the PRD unless explicitly asked to.
   * Prefer working on items that tie into BACKLOG.md or open issues.

3. **Avoid monochat:**
   * Assume this chat/agent session is temporary.
   * Anything important must be written back into the repo.

### 5.2. During Work: Use "Non-Build" Turns to Sync Memory

We distinguish between:

* **Build turns** - implementing or editing code/tests/configs that must be correct and atomic.
* **Non-build turns** - reflection, summarization, planning, updating docs.

**Approximate rule of thumb:**

> For every ~5 build turns, use **at least 1 non-build turn** to:
>
> * Update `DECISIONS.md` with new decisions,
> * Update `BACKLOG.md` if the plan changed,
> * Add/adjust entries in `GLOSSARY.md`,
> * Or write a short summary in the delta log.

This keeps the memory bank from drifting out of sync with the actual work.

### 5.3. After Significant Changes

If you:

* Add a major feature,
* Refactor a core module,
* Change a public API,
* Change how the reports/metrics work,

...then you should:

1. Add an entry to `DECISIONS.md`.
2. Update `PRD.md` or `BACKLOG.md` if the scope/roadmap changed.
3. If appropriate, update `GLOSSARY.md` with new terms.
4. Optionally add a delta log entry to `PRD.md` ("what we learned, what broke, what's next").

---

## 6. Rules for Human Collaborators

### 6.1. Treat Memory Docs Like Code

* **Changes go through PRs**, same as code.
* Reviewers should:
  * Check that decisions recorded in `DECISIONS.md` match the actual implementation.
  * Flag missing updates (e.g., big refactor but no decisions log).

### 6.2. When Opening a PR

Ask yourself:

* Did I make any **non-obvious tradeoffs**? -> Add to `DECISIONS.md`.
* Did the **shape of the product** change (scope, UX, metrics)? -> Update `PRD.md` or `BACKLOG.md`.
* Did we introduce new jargon or concepts? -> Update `GLOSSARY.md`.

### 6.3. When Onboarding Someone New

Point them first to:

1. `memory_bank/MASTER_MEMORY.md` (this file),
2. `memory_bank/INDEX.md`,
3. `DECISIONS.md` (at least the last few entries),
4. `PRD.md` and `BACKLOG.md`.

This should be enough for them to get the mental model of HaoLine and how we work.

---

## 7. Collaboration & "Master Version" in Git

Because the memory bank lives in the repo:

* **The master version is the main branch.**
* All updates are versioned and reviewable.
* Tools like Cursor/GitHub Copilot/ChatGPT can all be pointed at the **same canonical docs**.

If you're running agents in different environments (local Cursor, remote CI, etc.):

* They should all be instructed (via their own config/system prompts) to:
  * Read `memory_bank/INDEX.md` at start,
  * Follow `CURSOR_AGENT_RULES.md`,
  * Prefer updating memory docs instead of inventing new ones unless necessary.

---

## 8. Example Agent Workflow

1. Developer asks Cursor agent: *"Add a panel to the HaoLine UI showing estimated GPU cost per 1M tokens processed."*
2. Agent:
   * Reads `memory_bank/INDEX.md` -> opens PRD, BACKLOG, DECISIONS.
   * Designs change according to existing product goals.
   * Implements code + tests.
3. On a non-build turn, agent:
   * Adds an entry to `DECISIONS.md` documenting the cost metric chosen and why.
   * Updates `BACKLOG.md` to mark the related item as in progress / done.
4. Developer reviews both code and memory updates in the PR.

Result: anyone opening the repo later understands **what** changed and **why**, without digging through an old chat log.

---

*End of MASTER_MEMORY.md*

