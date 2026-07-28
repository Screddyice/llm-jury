# llm-jury

## Project Overview
<!-- Describe what this project does -->

## Tech Stack
- Python

## Common Commands


## Session Startup Protocol
On every session start:
1. Working context is auto-injected at session start (features, decisions, failures, rules)
2. Run `/claude-harness:start` for full context refresh with GitHub sync (optional)
3. Check `.claude-harness/features/active.json` for current priorities

## Development Rules
- Work on ONE feature at a time
- Always run /claude-harness:checkpoint after completing work
- Run tests before marking features complete
- Commit with descriptive messages
- Leave codebase in clean, working state

## Testing Requirements
<!-- Add your test commands -->
- Build: `npm run build`
- Lint: `npm run lint`
- Test: `pytest`
- Typecheck: `npx tsc --noEmit`

## Progress Tracking
See: `.claude-harness/sessions/{session-id}/context.json` and `.claude-harness/features/active.json`

## Memory Architecture (v3.0)
- `sessions/{session-id}/` - Current session context (per-session, gitignored)
- `memory/episodic/` - Recent decisions (rolling window)
- `memory/semantic/` - Project knowledge (persistent)
- `memory/procedural/` - Success/failure patterns (append-only)
- `memory/learned/` - Rules from user corrections (append-only)

