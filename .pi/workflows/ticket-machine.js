export const meta = {
  name: 'ticket-machine',
  description: 'Work the ready-for-agent frontier autonomously: per ticket, coder implements, reviewer approves, issue closes',
  phases: [{ title: 'Discover' }, { title: 'Build' }],
}

// Verdict the reviewer returns (schema-validated)
const VERDICT = {
  type: 'object',
  properties: {
    approved: { type: 'boolean' },
    findings: { type: 'array', items: { type: 'string' } },
  },
  required: ['approved', 'findings'],
}

// Frontier the dispatcher returns (schema-validated)
const TICKETS = {
  type: 'object',
  properties: {
    tickets: {
      type: 'array',
      items: {
        type: 'object',
        properties: { number: { type: 'number' }, title: { type: 'string' } },
        required: ['number', 'title'],
      },
    },
  },
  required: ['tickets'],
}

const REPO = 'xaltmiles/thor-monitor'
const MAX_FIX_ROUNDS = args?.maxFixRounds ?? 2
const MAX_TICKETS = args?.maxTickets ?? 20

const closed = []
const stuck = []
let processed = 0

while (processed < MAX_TICKETS) {
  phase('Discover')
  const frontier = await agent(
    `In this working directory, find the issue-tracker frontier for ${REPO}: open issues labeled ready-for-agent whose blocking issues are all closed. ` +
      `Follow the frontier query conventions in docs/agents/issue-tracker.md using the gh CLI. ` +
      `Return every such issue (number + title), in tracker order. If none remain, return an empty list.`,
    { label: 'frontier', phase: 'Discover', schema: TICKETS }
  )

  const tickets = (frontier?.tickets ?? []).slice(0, MAX_TICKETS - processed)
  if (!tickets.length) break
  log(`frontier: ${tickets.map((t) => '#' + t.number).join(', ')}`)

  // Sequential on purpose: one shared working tree, one local model server.
  // For parallel tickets, give coder agents isolation: 'worktree' and merge branches after.
  for (const t of tickets) {
    processed++
    phase(`#${t.number}`)

    const impl = await agent(
      `Implement issue #${t.number} ("${t.title}") end-to-end per your workflow: ` +
        `bootstrap from AGENTS.md, fetch the ticket and its parent spec, implement test-first, ` +
        `run the full suite, commit with a clear message. ` +
        `Do NOT close the issue — an independent review comes first.`,
      { label: `code:${t.number}`, agentType: 'coder', phase: `#${t.number}` }
    )
    if (impl === null) {
      log(`#${t.number}: coder failed or was skipped — leaving open`)
      stuck.push(t.number)
      continue
    }

    // Fresh reviewer each time (resume cannot combine with schema)
    let verdict = await agent(
      `Review the work just committed for issue #${t.number} of ${REPO}: fetch the ticket and parent spec, ` +
        `inspect the latest commits and changed files, run the suite if in doubt. Report a verdict.`,
      { label: `review:${t.number}`, agentType: 'reviewer', phase: `#${t.number}`, schema: VERDICT }
    )

    let round = 0
    while (verdict && !verdict.approved && verdict.findings?.length && round < MAX_FIX_ROUNDS) {
      round++
      await agent(
        `The reviewer found issues in #${t.number}: ${verdict.findings.join(' | ')}. ` +
          `Address them, run the full suite, commit. Do not close yet.`,
        { label: `fix:${t.number}-r${round}`, resume: `code:${t.number}`, phase: `#${t.number}` }
      )
      verdict = await agent(
        `Re-review the latest commits for issue #${t.number} after the fixes.`,
        { label: `re-review:${t.number}-r${round}`, agentType: 'reviewer', phase: `#${t.number}`, schema: VERDICT }
      )
    }

    if (verdict?.approved) {
      await agent(
        `Review approved. Close issue #${t.number} with a concise summary comment: what was built, key decisions, test results.`,
        { label: `close:${t.number}`, resume: `code:${t.number}`, phase: `#${t.number}` }
      )
      closed.push(t.number)
      log(`#${t.number} implemented, reviewed, closed`)
    } else {
      log(`#${t.number}: not approved after ${round} fix round(s) — left open for the human`)
      stuck.push(t.number)
    }
  }
}

return { closed, stuckOpen: stuck, ticketsProcessed: processed }
