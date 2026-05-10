Run a design spike: research, PoC, decisions, and proposed JIRAs

You are starting a spike for a feature in the Lightspeed Core project.

Follow the process in `docs/contributing/howto-run-a-spike.md`.  Use the
templates it references.

## Configuration

Read `docs/contributing/feature-design.config` (team policy) and, if
present, `.feature-design.config.local` at the repo root (per-developer
overrides). The local file's settings override the team file's; keys not
set in the local file fall through.

At the start of the session, announce the configuration in the manner
specified by `config_announcement_at_start`:
- `silent`: no announcement.
- `notable` (default): list which config files were read; list any
  actively-honored non-default settings relevant to the spike workflow
  (e.g., reviewers, default PoC ambition, decision presentation cadence);
  list any settings changed since the last session, computed against the
  state file at `config_state_file` (compare mtime + content hash).
- `full`: print the resolved (team + local) config in full.

After announcing, update the state file with the current mtime and hash
of each config file.

If `agent_may_suggest_config_changes_during_work` is `yes`, you may
propose one-shot or persistent config changes mid-session when they would
help the current work — but always ask the user before persisting.

## Fetching the JIRA ticket

If the user provides a JIRA ticket number (e.g., "1234" or "LCORE-1234"),
fetch the ticket content by running `sh dev-tools/fetch-jira.sh <number>`.
The output includes child issues — decide which linked tickets to fetch
for additional context.

Pass `--comments` (e.g., `sh dev-tools/fetch-jira.sh --comments 1234`)
when ticket comments may carry decisions or context not present in the
description (common for older feature tickets where the spike scope was
negotiated in comments). Off by default to keep output short.

Pass `--linked-depth N` (e.g.,
`sh dev-tools/fetch-jira.sh --linked-depth 1 1234`) to recurse N levels
into subtasks, linked issues, and parent-relation children. `N=1` is
typically what you want at spike kickoff: fetches the requested feature
ticket plus all its immediate Epics, subtasks, and blocking/blocked-by
issues in one call. Capped at 3. Cycle-safe (keys seen via multiple
paths are fetched once). Off by default (depth 0).

Otherwise, the user will provide context about the feature directly.

## Branch and working tree

Before creating the spike branch:

1. Confirm the proposed branch name with the user (per
   `branch_name_pattern`).
2. Check the working tree. If it's not clean, honor `on_dirty_working_tree`:
   - `ask` (default): describe the situation and ask the user whether to
     use a worktree (per `worktree_path_pattern`), move untracked files
     aside (per `move_aside_path_pattern`), stop and let the user clean
     up manually, or proceed anyway. Wait for an answer before acting.
   - `worktree` / `move_aside` / `stop`: act per the policy.
3. Create the branch off `branch_off` (default `upstream/main`) after
   confirmation.

## Drafting the spike doc

When proposing JIRAs in the spike doc, specify the type for each ticket
using `<!-- type: Task -->` or `<!-- type: Story -->` (see the JIRA ticket
template).  Include the e2e kickoff Story and step-definitions counterpart
Task as the first two proposed JIRAs by default (per
`require_e2e_kickoff_jira` and `require_e2e_step_definitions_jira`).

Use `dev-tools/file-jiras.sh --help` for filing details.

## Decision presentation

Honor `decision_presentation_cadence`:
- `per_decision`: ask the user at every decision before moving on.
- `batch`: present all decisions in one summary near the end and let the
  user confirm/override the batch.
- `mixed` (default): per-decision for strategic decisions; batched for
  technical decisions.

State a confidence value with each recommendation (per
`require_confidence_per_recommendation` and `confidence_format`).

## Use task tracking

Use `TaskCreate` / `TaskUpdate` to track the spike phases (set up →
research → design alternatives → PoC → spike doc → spec doc → JIRAs).
Spikes are multi-phase; tracking improves continuity and reviewability.

## Keep spike doc and spec doc coupled

Treat the spike doc and the spec doc as a coupled pair. Whenever you make
a non-trivial update to one — a decision recommendation changes, a PoC
finding is integrated, the user overrides a default, a reviewer comment
is incorporated — **check whether the other needs a corresponding
update**. If yes, propose the update in the same turn.

## End-of-session reminder

When the spike PR is opened (step 7 of the howto), remind the user that
`/spike-finalize` exists and what conditions trigger it: decisions
confirmed by reviewers, JIRAs filed via `/file-jiras`, ready to merge the
spike PR.
