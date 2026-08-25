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
| E | ApprovalService abstraction | MERGED | #1274 (`cd1f9782`) | Neutral `ApprovalAnswerer` seam, closed vocabulary, wire view; cancellation currently checked before/after synchronous answerer |
| F | DeepSeek Harness bridge | MERGED | #1277 (`bcddddf9`) | Base bridge merged; request-id/scope/justification correlation repair tracked in #1287 |
| G | Safe child-agent delegation | MERGED | #1275 (`fcb8c9da`) | Monotonic `DelegationContext`; finite resource-ceiling regression repair tracked in #1286 |
| H-1 | CCOS Enterprise identity/RBAC/isolation primitives | MERGED | #1278 (`babdc2b7`) | TenantId/OrgId/ProjectId/WorkspaceId, tenant-scoped rules, workspace path isolation |
| H-2 | Scoped secret capability store | MERGED | #1279 (`e6bdfeae`) | Opaque handles, explicit grants, revocable, audit views never contain values |
| H-3 | Resource-budget + explicit-egress policy primitives | MERGED | #1280 (`80c7d694`) | Fail-closed capability checks exist; real OS/network enforcement is not yet wired into ToolRuntime |
| H-4 | In-memory correlated enterprise audit trail | MERGED | #1281 (`bbae8850`) | SHA-256 chained tenant..artifact correlation |
| H-6 | Durable enterprise audit + automatic runtime emission | OPEN PR | #1325 | `FileEnterpriseAuditTrail` JSONL chain verified on read, shared `EnterpriseAuditSink`, and automatic `DeepSeekBridge` runtime emission |
| H-5 | Kernel-enforced resource governance | OPEN PR | `feat/sciagent-real-resource-enforcement` | honest fail-closed enforcement: inherited `RLIMIT_FSIZE` plus deny-all INET egress via seccomp-BPF/bwrap network namespace; tree-wide memory/process/CPU/wall-time, GPU caps and host allow-lists are refused until a non-escapable backend exists |
| H-7 | Single authoritative approval-policy source | OPEN PR | `feat/unified-policy-source` | `SharedApprovalPolicy` cell owned by PermissionGate, service-bound via `bind_to_gate`/`with_shared_policy`; poison ⇒ Never everywhere, writes propagate both ways |
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
| Approval Ask/Never | `ApprovalPolicy` in session events, last wins | `ApprovalPolicy::Ask/Never`; `ApprovalService::bind_to_gate` binds the service to the gate's single `SharedApprovalPolicy` cell, so both layers enforce one value | none remaining at source level: divergence between gate and service is structurally impossible once bound; owned mode preserved for standalone services | a poisoned shared cell fails closed (`Never` presentation, pre-answerer rejection) | this branch | #1270/#1274 | live-switch, write-back, poison and durable-replay tests + CI |
| Session durability | append-only `approval/policy` session event, replay last wins | `FileApprovalPolicyStore` JSONL SHA-256 chained, strict 0..n | SciRust adds hash-chain corruption detection | corrupt log cannot grant more privilege | #1271 | #1270 | tests restart/torn-tail |
| ApprovalRequestId | `Branded<'ApprovalRequestId'>`, service-issued per request | 128-bit validated strong type, CSPRNG+counter | base bridge did not preserve the generated id across its wire lifecycle; repair in #1287 | supervision events must correlate Requested/Resolved exactly | #1287 OPEN | #1272/#1274/#1277 | same non-empty id across request/resolution/audit |
| request/cancellation race | cancellation can win asynchronously; late result discarded | synchronous `ApprovalAnswerer`; token checked before and after `answer()` | true interruptible/async cancellation is still TODO | cancellation cannot currently interrupt a blocked synchronous answerer | TODO atomic cancellation PR | #1274 | blocking-answerer cancellation test |
| audit pairing | `approval/asked` + `approval/decided` same `id` | `ApprovalAuditEvent` Requested/Resolved same `request_id` | audit layer pairs correctly; bridge correlation repair in #1287 | pairing integrity | #1273/#1287 | test pair same id |
| session replay | durable session events replayed | `FileApprovalAudit` + `FileApprovalPolicyStore` | equivalent at policy/audit-store level | restart reproduces recorded authority | #1271/#1273 | #1270 | tests restart |
| model-facing policy context | runtime-context snapshot, live switch notices | `DeepSeekBridge::approval_policy()` reads the bound shared cell — the same source enforcement reads; live switches are visible to the model without re-binding | live SWITCH NOTICES to the model (proactive notification) remain harness-side only | enforcement and presentation cannot disagree about Ask/Never | this branch | #1277 | bridge live-switch test |
| tool-call correlation | `callId?: CallId` | `call_id` String preserved independently | equivalent | traceability | #1272 | — | tests |
| sandbox escalation | one-shot escalation in approval request | one-shot escalation + independent approval requirement | equivalent | escalation cannot bypass approval | #1266 (MERGED) | — | tests |
| child delegation | `source: 'delegation'` seeds override into child | `DelegationContext` ceilings for tools/sandbox/resources/secrets/workspace | finite optional resource ceiling bug identified; repair in #1286 | child must never remove a finite parent resource ceiling | #1286 OPEN | #1275 | finite-parent/None-child regression tests |
| fail-closed behavior | missing answerer => unavailable | missing/erroring answerer => Unavailable | equivalent for answerer availability | no silent grant | #1274 | — | tests |
| enterprise budgets/egress | n/a | fail-closed governance wired through `ToolRuntime::execute_governed`; currently enforceable claims are inherited `RLIMIT_FSIZE` and deny-all INET egress via seccomp/bwrap isolation | tree-wide memory/process/CPU/wall-time need cgroup/PID-namespace-class control; GPU caps and per-host egress allow-lists also refuse rather than approximate | declared limits must correspond to non-escapable enforcement | H-5 branch | #1280 | live tests: SIGXFSZ via RLIMIT_FSIZE and bash `/dev/tcp` blocked by seccomp; capability tests prove unsupported tree-wide limits refuse before execution |
| enterprise correlated audit | n/a | `FileEnterpriseAuditTrail` JSONL SHA-256 chain verified on every read + `EnterpriseAuditSink`; `DeepSeekBridge::with_enterprise_audit` emits correlated executed/failed/rejected calls | rotation and independently constructed/cross-process concurrent writers remain a single-writer limitation | restart preserves the trail; audit refusal prevents an unaudited success from reaching the model | #1325 | #1281 | restart/tamper/torn-tail/concurrency tests; bridge emission + fail-closed sink tests |

## Hardware evidence/status

- Physical Jetson Thor execution is available and has produced accepted evidence; it is not globally `HARDWARE PENDING`.
- M54 full-model prefill evidence recorded by #1284: run `32308929421`, exact head `e7640173cf3c9680710683753f24a5d98753cdf4`, physical runner `tarek-scirust-arm64-01`, NVIDIA Thor/Vulkan; prompt-128 and prompt-512 portable/vec4 final logits were bit-identical. No default-routing performance promotion was made.
- Separate CUDA production-gate runs have reached real Thor execution and exposed a cached-vs-naive decode parity failure. Treat that as an unresolved correctness issue, not as generic runner unavailability.
- `HARDWARE PENDING` should be used only for a specific test that has not actually executed on the required hardware.
- `BLOCKED`: none globally; individual hardware jobs may still be deferred by runner/GPU occupancy.
- `UPSTREAM DIFFERENCE`: SciRust persistent grants (Session/Always) go beyond DeepSeek's one-shot `allowed-once`; DeepSeek has no equivalent persistent-grant model. Do not invent implicit revocation.
