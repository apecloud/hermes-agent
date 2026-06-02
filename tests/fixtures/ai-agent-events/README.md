# Hermes Runtime Raw Event Base Cases

This canonical Hermes Agent repo fixture directory freezes Hermes Runtime event streams used as regression fixtures for Cloud projection and Console UI replay.

## Policy

- Treat `cases/*/events.jsonl` as base Runtime event fixtures.
- Refresh base raw events only when Hermes Agent Runtime code or event contract changes.
- Cloud/API/frontend changes should update assertions, expected outputs, or test results, not the raw base events.
- Fixtures preserve Runtime event structure and ordering but redact kubeconfig paths, auth tokens, passwords, secrets, and API keys before storage.
- Candidate partial replays live under `candidates/` and must not be promoted to base until captured complete from event id 1.

## Layout

- `manifest.json`: case index, update policy, pending coverage gaps.
- `cases/<case-id>/events.jsonl`: base raw Hermes Runtime events.
- `cases/<case-id>/metadata.json`: prompt, org/cluster, run IDs, runtime/cloud version notes, event counts.
- `cases/<case-id>/assertions.json`: expected behavior for backend/API/frontend tests.
- `candidates/<case-id>/events.partial.jsonl`: useful evidence that is not complete enough to be a base fixture.

## Current Base Cases

1. `normal-diagnosis-yam23-20260601-132718`
   - Covers ordinary diagnosis with `reasoning.available`, multiple tool phases, `agent.step`, `message.delta`, and `run.completed`.
   - Source: previous full capture `/Users/alal/kubeblocks-console-e2e/output/hermes-runtime/hermes-raw-events-20260601-132718.jsonl`.
2. `quick-status-yam23-20260602-150122`
   - Covers a fresh real Hermes Agent Runtime conversation against cloud-dev yam23 after the reasoning/progress contract work.
   - Complete capture from event id 1 through `run.completed`; useful as the current canonical quick status baseline for reasoning delta, tool progress, and final answer replay.

## Current Candidate Evidence

- `long-tool-call-yam23-20260602-134148-partial`: owner-reported perceived stuck run; useful for task #90 analysis, but Runtime replay starts at event id 472 so it is incomplete.
- `long-tool-call-yam23-20260602-140657-partial`: follow-up turn from same conversation; also replay-truncated.

## Pending Coverage

See `manifest.json -> pendingCases`. P0 gaps are long-tool-call complete capture, approval approved/rejected, and concrete tool failure.

## External consumers

Cloud apiserver and Console tests should reference or copy these fixtures from the Hermes Agent repo. Do not treat `kubeblocks-console-e2e/output/hermes-runtime/base-cases` as canonical; it is only the staging copy used when this first fixture set was assembled.
