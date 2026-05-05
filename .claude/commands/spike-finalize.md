Finalize a spike after its PR has been merged: replace placeholders, verify drift, prompt about Google Docs

You are finalizing a spike whose PR has been merged. The spike doc and spec
doc now live in the repo permanently; PoC code and evidence have been
removed. Implementation JIRAs have been filed under the parent feature
ticket.

Follow the process in `docs/contributing/howto-run-a-spike.md`, step 11
("Finalize after merge").

## Configuration

Read `docs/contributing/feature-design.config` and (if present)
`.feature-design.config.local`. Honor:
- `auto_replace_jira_placeholders`
- `verify_spike_spec_doc_drift`
- `verify_no_orphan_jira_references`
- `remind_about_google_doc_after_merge`
- `config_announcement_at_start`

Announce config files read at session start, per
`config_announcement_at_start` (same behavior as `/spike`).

## Inputs

The user will identify the feature, e.g., by directory name
(`docs/design/<feature>/`) or by spike ticket key. From this:
- Locate the spike doc (`<feature>-spike.md`) and spec doc (`<feature>.md`).
- Locate the parsed JIRAs directory (`docs/design/<feature>/jiras/`) if
  it exists.

## Steps

### 1. Replace LCORE-???? placeholders

If `auto_replace_jira_placeholders = yes`:

- For each filed ticket file in `docs/design/<feature>/jiras/`, read the
  `<!-- key: LCORE-XXXX -->` metadata.
- Find the corresponding `LCORE-????` placeholder in the spike doc by
  matching the ticket title (`### LCORE-???? <title>`) to the filed
  ticket's title.
- Replace `LCORE-????` with the real key in the spike doc.
- Show a diff to the user before writing.

If a placeholder cannot be matched to a filed ticket, list the unmatched
items and ask the user to resolve manually.

### 2. Verify spike ↔ spec doc don't drift

If `verify_spike_spec_doc_drift = yes`:

- Read both docs.
- Identify decisions in the spike doc that should be reflected in the spec
  doc (look at "Strategic decisions", "Technical decisions",
  "Stakeholder decisions" sections).
- Cross-check that the spec doc's Requirements / Architecture /
  Implementation Suggestions reflect the *resolved* decisions (i.e., the
  recommendations the reviewers accepted).
- Report any apparent drift to the user — don't fix automatically;
  document drift requires human judgment.

### 3. Verify no orphan JIRA references

If `verify_no_orphan_jira_references = yes`:

- Grep for `LCORE-????` in the spike doc and the spec doc. Any remaining
  matches are unresolved placeholders.
- Grep for `LCORE-` references; for each found key, confirm the ticket
  exists (via `dev-tools/fetch-jira.sh <key>` if needed). Report any that
  return errors.

### 4. PoC artifacts cleanup check

Confirm `docs/design/<feature>/poc-results/` was removed before merge (per
`remove_poc_dir_before_merge` and the howto step 10). If it still exists,
remind the user; do not delete automatically.

### 5. Google Docs reminder

If `remind_about_google_doc_after_merge = yes`:

Print a reminder, in this shape (adapt wording as appropriate):

> Now that the spike+spec PR has merged, consider creating Google Docs
> versions of:
>   - the spike doc at `docs/design/<feature>/<feature>-spike.md`
>   - the spec doc at `docs/design/<feature>/<feature>.md`
>
> Google Docs versions are easier for the wider team to read and comment
> on. Markdown → Google Doc conversion: copy the rendered Markdown into
> a new Doc, or use a converter; either way, link the resulting docs
> back from the JIRA feature ticket so they're discoverable.

Do NOT attempt to create the Google Docs yourself; this is a manual
step the user takes.

### 6. Summary

Print a short end-of-session summary listing what was changed
(placeholder replacements, files updated), what was flagged (drift,
orphan refs, missing cleanup), and what's left for the user (Google Docs
creation, any flagged items needing manual resolution).
