# Skeptic checklist (generic)

Use this checklist when you review an AI-produced claim, hypothesis, or extracted knowledge graph.

## Evidence & traceability
- Is there a **primary source** (paper, dataset, protocol) cited for each key claim?
- Is the citation **locatable** (DOI/PMID/arXiv + page/section/figure/table)?
- Are we mixing **review articles** with **primary evidence**? If yes, mark it.

## Scope & assumptions
- What is the **population/system** studied (organism, material system, dataset, domain)?
- What are the **boundary conditions** (temperature, concentration, hardware, hyperparameters, etc.)?
- Are there hidden assumptions (e.g., simplified model, idealized conditions, selection bias)?

## Time & causality
- Is the claim time-dependent (before/after, longitudinal, temporal ordering)?
- If causality is claimed: is it supported by design (RCT, ablation, IV, mechanistic evidence), or only correlation?

## Methods quality
- Sample size / power (or dataset size) — is it adequate?
- Controls / baselines — are they appropriate?
- Metrics — do they match the claim? Any proxy metrics that might mislead?

## Conflicts & uncertainty
- Are there contradictory findings across sources? If yes, list them explicitly.
- What is the uncertainty / confidence level? What would falsify the claim?

## Reproducibility
- Are code/data/protocols available?
- If not, what minimal details are needed to reproduce?

## Next actions
- What is the **single best follow-up**: additional paper, experiment, analysis, or expert review?
