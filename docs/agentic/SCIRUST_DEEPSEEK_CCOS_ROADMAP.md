# SciRust / SciAgent / CCOS — DeepSeek-Interoperable Agentic Runtime Roadmap

Status vocabulary (Section 27 of the program):

- `MERGED` — merged into `master`, evidence in CI/local validation
- `OPEN PR` — branch pushed, PR open, CI in progress
- `LOCAL` — implemented in a local worktree, not yet pushed
- `TODO` — designed, not implemented
- `BLOCKED` — external blocker (hardware/credentials/service)
- `UPSTREAM DIFFERENCE` — SciRust intentionally diverges from DeepSeek Harness
- `VALIDATED` — passed the listed validation
- `HARDWARE PENDING` — requires Jetson Thor / CUDA hardware, not yet executed

Reference upstream: DeepSeek Harness `master` @ `141eb6fef83422698aef7a981029e843e8161534`
(inspected `packages/interaction/user-approval/src/index.ts` and `types.ts`).

## Architecture target

```
DeepSeek Harness -> CCOS Enterprise -> Other agent clients
        -> Versioned SciRust Agent Protocol
        -> Durable Session Runtime
        -> Durable Approval Policy/Audit
        -> ToolRuntime -> PermissionGate -> ApprovalService
        -> Sandbox Permission/Escalation
        -> Bubblewrap / Landlock
        -> SciRust Scientific Capabilities
        -> Verification / Attestation / Provenance
        -> Evidence-driven Optimization/Research
```

No external frontend may bypass this chain. No agent may authorize itself.
No child may widen its parent's authority.

## Roadmap

| Phase | Capability | Status | PR / SHA | Notes |
|-------|-----------|--------|----------|-------|
| A | ApprovalPolicy Ask/Never | MERGED | #1270 (`c57c7ecd`) | `Ask` default, `Never` rejects before approver, fail-closed, shared across clones |
| B | Durable approval policy per session | MERGED | #1271 (`ec2a186b`) | `FileApprovalPolicyStore` JSONL + SHA-256 chain, last-valid-event wins, fail-closed replay |
| C | ApprovalRequestId | MERGED | #1272 (`a7436775`) | 128-bit strong id, independent from call_id, serde round-trip, CSPRNG + counter |
| D | Durable approval audit / session event store | MERGED | #1273 (`1e809e84`) | `FileApprovalAudit`, append-only, SHA-256 chained, pairing, replay, fail-closed |
| E | Complete ApprovalService abstraction | OPEN PR | #1274 (`cd1f9782`) | Neutral `ApprovalAnswerer` seam, cancellation races, closed vocabulary, wire view |
| F | DeepSeek Harness bridge | OPEN PR | #1277 (`00141db5`) | `deepseek_bridge.rs`, tool defs/calls, approval, session, streaming events, error protocol |
| G | Safe child-agent delegation | MERGED in E | #1275 (`fcb8c9da`) | Monotonic `DelegationContext`, child ⊆ parent ceilings, nested preserved |
| H-1 | CCOS Enterprise identity/RBAC/isolation | OPEN PR | #1278 (`5a28c655`) | TenantId/OrgId/ProjectId/WorkspaceId, tenant-scoped rules, workspace path isolation |
| H-2 | Scoped secret capability store | OPEN PR | #1279 (`14b15f52`) | Opaque handles, explicit grants, revocable, audit views never contain values |
| H-3 | Resource budgets + explicit egress | OPEN PR | #1280 (`0063a9a3`) | EgressPolicy deny-all default, ResourceBackend capability seam, fail-closed |
| H-4 | Correlated enterprise audit trail | OPEN PR | #1281 (`7c6ad646`) | SHA-256 chained, tenant..artifact correlation, tamper-evident |
| I | Scientific autonomy (generalize optimizer) | PARTIAL | PR #1254 MERGED | CPU/SIMD/CUDA/WGPU generalization TODO |
| — | Evidence-driven optimization loop | MERGED | #1254 (`c55dcc04`) | baseline -> generate -> compile -> verify -> benchmark -> promote |
| — | Typed ToolRuntime contracts | MERGED | #1256 | typed ToolCall, registry, validation, policy hooks |
| — | Bubblewrap sandbox seam | MERGED | #1257 | |
| — | Native Landlock fallback | MERGED | #1260 | |
| — | Default sandbox workspace-write | MERGED | #1261 | |
| — | Per-call PermissionGate | MERGED | #1262 | Allow/Ask/Deny, DENY > ASK > ALLOW > fallback |
| — | Session-scoped approval grants | MERGED | #1264 | |
| — | Typed one-shot outcomes | MERGED | #1265 | allowed-once / rejected / cancelled / unavailable |
| — | One-shot sandbox escalation | MERGED | #1266 | read-only / workspace-write / danger-full-access |
| — | Structured approval audit lifecycle | MERGED | #1267 | bounded in-memory journal |
| — | Persistent approval scope | MERGED | #1268 | Once / Session / Always / Decline |
| — | Clippy 1.89 CI compatibility | MERGED | #1269 (`32842ffc`) | |
| — | Agentic runtime roadmap doc | MERGED | #1276 (`d36d1399`) | this document |

## Gap analysis vs DeepSeek Harness (SHA `141eb6fe`)

| Capability | DeepSeek current behavior | SciRust current behavior | Gap | Security impact | Proposed atomic PR | Dependency | Validation |
|-----------|--------------------------|--------------------------|-----|-----------------|--------------------|------------|------------|
| Approval Ask/Never | `ApprovalPolicy` in session events, last wins | `ApprovalPolicy::Ask/Never`, shared Arc<RwLock>, Never fails closed | none functionally; SciRust adds explicit Never semantics | Never blocks new approvals deterministically | #1270 (MERGED) | — | local tests + CI |
| Session durability | append-only `approval/policy` session event, replay last wins | `FileApprovalPolicyStore` JSONL SHA-256 chained, strict 0..n | SciRust adds hash-chain corruption detection | corrupt log cannot grant more privilege | #1271 | #1270 | tests restart/torn-tail |
| ApprovalRequestId | `Branded<'ApprovalRequestId'>`, service-issued per request | 128-bit validated strong type, CSPRNG+counter | SciRust adds wire validation (parse rejects malformed) | no id spoofing by answerer | #1272 | — | tests parse/generate/concurrency |
| request/cancellation race | cancellation can win; late result discarded | `CancellationToken`, check before+after answerer | equivalent | no late grant | #1274 | #1272 | test late-answer-discarded |
| audit pairing | `approval/asked` + `approval/decided` same `id` | `ApprovalAuditEvent` Requested/Resolved same `request_id` | equivalent | pairing integrity | #1273 | #1272 | test pair same id |
| session replay | durable session events replayed | `FileApprovalAudit` + `FileApprovalPolicyStore` | equivalent | restart reproduces authority | #1271/#1273 | #1270 | tests restart |
| model-facing policy context | runtime-context snapshot, live switch notices | `approval_policy()` on gate; wire view carries policy | SciRust exposes via gate, not yet a model-context snapshot | model sees effective policy | #1274 | #1270 | test wire view |
| tool-call correlation | `callId?: CallId` | `call_id` String preserved independently | equivalent | traceability | #1272 | — | tests |
| sandbox escalation | one-shot escalation in approval request | one-shot escalation + independent approval requirement | equivalent | escalation cannot bypass approval | #1266 (MERGED) | — | tests |
| child delegation | `source: 'delegation'` seeds override into child | `DelegationContext` monotonic ceilings, child ⊆ parent | SciRust richer: tools/sandbox/resources/secrets/workspace ceilings | child cannot widen parent | #1275 | #1274 | tests nested/termination |
| fail-closed behavior | missing answerer => unavailable | missing/erroring answerer => Unavailable | equivalent | no silent grant | #1274 | — | tests |

## Notational statuses

- HARDWARE PENDING: SciAgent Thor gate (`sciagent-thor-gate.yml`, CUDA parity + 304M benchmark) and FLAT M33 resident decode benchmark run on self-hosted Jetson Thor runners; queued when no runner is free. No CUDA result is claimed from CPU-only runs.
- BLOCKED: none currently.
- UPSTREAM DIFFERENCE: SciRust persistent grants (Session/Always) go beyond DeepSeek's one-shot `allowed-once`; DeepSeek has no equivalent persistent-grant model. Do not invent implicit revocation.
