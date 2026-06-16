# Compliance Report: vllm-clinical-rag
Resource type: Guide
Evaluated: 2026-06-16 (re-scored after Guide conversion)

## Summary

**Compliance Status:** 14 of 14 required criteria passed (Universal 6/6 · Guide 8/8)

**Overall Assessment:** After the Guide conversion this resource now meets every *scored* quality criterion. The remaining work is not criteria failures but **publication-process gates** (hosting in the Anaconda-Labs org, registering it, and a real end-to-end test to make the "Active" badge truthful). Strong engineering plus secure-by-default discipline throughout.

## Strengths

- **Secure-by-default, verifiable.** No secrets (`.env` gitignored, `.env.example` present), conda-only, no pip, no HuggingFace Hub. Environment now pinned to exact `main`-channel versions.
- **Real teaching structure.** Learning objectives, knowledge + install prerequisites, per-section start-state and checkpoints, sample outputs, and an extension challenge.
- **A documented failure mode.** The "Known issues" section shows a real error + root cause + fix (satisfies the recommended `failure_mode` criterion).
- **Clean, coherent code** aligned with the notebook (no more stale BM25/vLLM references).

## Universal criteria

| Criterion | Result | Notes |
|-----------|--------|-------|
| readme_sections | PASS | Description + Audience + named owner all present. |
| named_owner | PASS | Federico Rubiano (@federicorubiano). |
| environment_spec | PASS | `environment.yml` / `environment-local.yml` now pinned exactly (verified `main`-channel versions). Recommend `conda-lock` to confirm the solve. |
| no_secrets | PASS | `.env` untracked + gitignored; `.env.example` present; no secret patterns. |
| no_pii | PASS | Public Merck text + synthetic queries. |
| license | PASS | MIT LICENSE at root; holder updated to Anaconda, Inc. |

## Guide-specific criteria

| Criterion | Result | Notes |
|-----------|--------|-------|
| learning_objectives | PASS | 5 measurable, action-verb objectives ("What you'll learn"). |
| prerequisites | PASS | Both knowledge and installation prerequisites, with links. |
| completion_time | PASS | 45–90 min estimate. |
| dependency_tier | PASS | External deps classified (Desktop Tier 3, Merck Tier 2, Platform Tier 3/optional). |
| starting_state | PASS | Each build step/section opens with a start-state line. |
| checkpoints | PASS | Each step ends with a ✅ checkpoint. |
| output_examples | PASS | Sample outputs in fenced blocks per checkpoint + notebook cells. |
| extension_challenges | PASS | Optional "Extension challenges" section. |

## Guide recommended (do not fail)

| Criterion | Result | Notes |
|-----------|--------|-------|
| glossary | Not present | Optional: add RAG/FAISS/embedding/chunk terms. |
| failure_mode | Present ✅ | "Known issues" section. |
| common_mistakes | Partial | Covered indirectly by Known issues. |
| ci_smoke_test | Not present | `test_api.py` exists; no GitHub Actions workflow. |

## Next steps

### Critical (blocks publication — process, not criteria)
- **Host in the Anaconda-Labs org** and **register in resource-registry** (now unblocked by org access).
- **Run end-to-end once** on a live Desktop setup and update "Last tested" so the **Active** badge is truthful.

### High impact
- Run `conda-lock` to confirm the pinned env solves and capture a lockfile.
- Re-run the notebook so the committed cells show **real** outputs (current samples are illustrative).
- Capture the Gradio screenshot (`screenshots/gradio-ui.png`).
- Update `anaconda-project.yml` — it still declares the old vLLM variables and unpinned packages (handled in HANDOFF for Claude Code).

### Polish (recommended, optional)
- Add a glossary, an explicit "common mistakes" callout, and a CI smoke-test workflow.

## Conclusion
The resource now passes all 14 scored criteria as a Guide. What remains is publication mechanics — get it into the Anaconda-Labs org, register it, and do one real end-to-end run — not content work.
