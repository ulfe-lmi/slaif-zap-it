# OpenCode OAP runbook

```text
CODING=$HOME/opencode-work/slaif-zap-it
STRATEGIC=$HOME/opencode-supervision/slaif-zap-it
```

Bootstrap: `bash oap/bin/bootstrap-two-opencode-oap.sh`. Runtime config is private
mode 0600 under strategic home. Set model IDs as `provider/model`; variants are
provider-specific. For one role use either MODEL plus optional VARIANT, or an
existing AGENT name, never both. Coding `opencode run` receives model/variant
flags directly. The strategic TUI launcher materializes model/variant into a
private generated OpenCode agent because the TUI has no `--variant` flag.
`permission=allow` plus launcher `--auto` provides unattended tool use;
protected-host law remains mandatory. OpenCode sharing is disabled.

Start visible two-pane session: `bash oap/bin/launch-tmux.sh`. Coding wrapper
blocks on FIFO outside OpenCode and starts a fresh `opencode run` for each order.
Strategic is persistent interactive TUI. No polling or two-minute sleep is used.

Recovery: restart only dead coding wrapper; tell existing strategic TUI it is
waiting; strategic reconciles active/GitHub/report and re-sends control `OK` only
if the round is unresolved. Never invent a new letter solely for process crash.

## HWP/HJP and `CRITICAL.md`

Human Work Preloading supplies architecture, constraints and roadmap before the
loop. Strategic therefore owns ordinary and consequential provisional decisions;
it must not stop merely because it would prefer a human choice. Human Judgment
Postloading defers only rare material adjudications. Apply all five conditions in
`CRITICAL.md`; normally the work order says `Deferred human adjudication: NONE`.

`CRITICAL.md` is intentionally absent from the always-loaded `opencode.json`
instruction lists. Strategic reads it at startup, before deployment/release, and
when a related dilemma arises; coding reads it only for an ordered append or
relevant cross-reference. This prevents the register from consuming every turn
or becoming a substitute for hard reasoning.

For an actual critical dilemma, strategic authors exact entry bytes and a same-PR
order. Coding appends them with:

```bash
python oap/bin/append_critical.py \
  --repo-root "$HOME/opencode-work/slaif-zap-it" \
  --source /path/to/strategic-authored-entry.md \
  --id CRIT-0001
```

Agents never edit or close prior entries. Before deployment/release, a human
appends `ACCEPTED` dispositions for all applicable gates. `DEFERRED`,
`REJECTED`, and `CHANGE REQUIRED` remain blocking.

Inspect state:

```bash
python oap/bin/check_state.py \
  --repo-root "$HOME/opencode-work/slaif-zap-it" \
  --strategic-home "$HOME/opencode-supervision/slaif-zap-it"
```
