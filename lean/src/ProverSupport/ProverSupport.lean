import Mathlib

open BigOperators Nat Finset

/-! # ProverSupport

Lean-native proof search for the agent.

`prover_finish` runs the hammer chain *inside* a single Lean invocation —
`grind`, `simp`, `ring`, `omega`, `linarith`, `nlinarith`, `norm_num`,
`decide`, `tauto`, `aesop`, `positivity` — trying each tactic until all goals
are closed. `grind` (Lean 4.33+) is tried first as the strongest hammer.
Tactic names are parsed at run time (via `runParserCategory`) so macros like
`tauto`/`aesop`/`grind` expand only when actually executed, never at quotation
time.

The agent previously drove this chain from Python by spawning `lake env lean`
once per hammer (~10 compiles per problem); `prover_finish` does the same
search in one compile, entirely inside Lean.
-/

namespace ProverSupport

open Lean Elab Tactic Meta

/-- The hammer chain, in order of preference (`grind` first on Lean ≥4.33).
`bv_decide` sits last-but-one: strong on Bool/BitVec goals, near-instant
rejection elsewhere. `trivial` mops up degenerate goals. -/
def hammerNames : List String :=
  ["grind", "simp", "ring", "omega", "linarith", "nlinarith", "norm_num",
   "decide", "tauto", "aesop", "positivity", "bv_decide", "trivial"]

/-- Try one hammer; goals are untouched when it fails or is unknown. -/
def tryHammer (name : String) : TacticM Unit := do
  let stx := Lean.Parser.runParserCategory (← getEnv) `tactic name
  match stx with
  | .ok s => try evalTactic s catch _ => pure ()
  | .error _ => pure ()

/-- Run the hammer chain until all goals are closed (or the chain is spent). -/
elab "prover_finish" : tactic => do
  for name in hammerNames do
    let goals ← getGoals
    if goals.isEmpty then
      return
    tryHammer name
    let goals' ← getGoals
    if goals'.isEmpty then
      return

/-- Are there no unsolved goals left? -/
def noGoals : TacticM Bool := do
  return (← getUnsolvedGoals).isEmpty

/-- Run `t`; on failure restore the state and return false.
On success return whether every goal was closed. -/
def closeWith (t : TacticM Unit) : TacticM Bool := do
  let saved ← saveState
  try
    t
    noGoals
  catch _ =>
    saved.restore
    pure false

/-- Identifiers of the goal's local variables, propositions first, data after. -/
def localVarIds : TacticM (List (Ident × Bool)) := do
  let g ← getMainGoal
  let decl := (← getMCtx).getDecl g
  let mut props : List (Ident × Bool) := []
  let mut data : List (Ident × Bool) := []
  for l in decl.lctx do
    if l.isLet then
      continue
    let isP ← isProp l.type
    let id := mkIdent l.userName
    if isP then
      props := (id, true) :: props
    else
      data := (id, false) :: data
  return props.reverse ++ data.reverse

/-- An `elimTarget` syntax referring to a local variable by name. -/
def elimTargetOf (id : Ident) : TSyntax `Lean.Parser.Tactic.elimTarget :=
  ⟨mkNode ``Lean.Parser.Tactic.elimTarget #[mkNullNode, id.raw]⟩

/-- Identifiers of the goal's *data* (non-proposition) variables. -/
def dataVarIds : TacticM (List Ident) := do
  return (← localVarIds).filterMap fun (id, isP) => if isP then none else some id

/-- Bounded native search: hammers, then case split / induction on local
variables, then `subst`, then `use` witnesses for existential goals, then
`simp_all` — every branch with full backtracking, depth-capped.
`budget` bounds the total number of search nodes so the tactic always
terminates regardless of the heartbeat limit. -/
partial def proverSearchDepth (budget : Nat) (depth : Nat) : TacticM Unit := do
  if budget == 0 then
    throwError "prover_search: budget exhausted"
  if ← noGoals then
    return
  if ← closeWith (evalTactic (← `(tactic| prover_finish))) then
    return
  if depth == 0 then
    throwError "prover_search: depth exhausted"
  -- make forall-bound variables into locals
  let _ ← closeWith (evalTactic (← `(tactic| intros)))
  -- B2: case split on every local variable (propositions first)
  for (id, _) in ← localVarIds do
    let et := elimTargetOf id
    if ← closeWith do
      evalTactic (← `(tactic| cases $et:elimTarget))
      proverSearchDepth (budget - 1) (depth - 1) then
      return
  -- B3: induction on data variables
  for id in ← dataVarIds do
    let et := elimTargetOf id
    if ← closeWith do
      evalTactic (← `(tactic| induction $et:elimTarget))
      proverSearchDepth (budget - 1) (depth - 1) then
      return
  -- B4: substitute along loop equations `x = t` / `t = x` with `x` a local var
  for (id, _) in ← localVarIds do
    let ok ← closeWith do
      evalTactic (← `(tactic| subst $id))
      proverSearchDepth (budget - 1) (depth - 1)
    if ok then
      return
  -- B5: witness search for existential goals (`use v`, `use v * w`, `use v + w`,
  -- plus triples `i * j * k` when there are few data variables)
  let target ← whnf (← (← getMainGoal).getType)
  if target.isAppOf ``Exists then
    let ids ← dataVarIds
    let mut candidates : Array (TSyntax `tactic) := #[]
    for id in ids do
      candidates := candidates.push (← `(tactic| use $id:ident))
    for (i, j) in ids.product ids do
      if i != j then
        candidates := candidates.push (← `(tactic| use $i:ident * $j:ident))
        candidates := candidates.push (← `(tactic| use $i:ident + $j:ident))
    if ids.length ≤ 3 then
      for (i, j, k) in (ids.product ids).flatMap (fun (i, j) => ids.map (fun k => (i, j, k))) do
        if i != j && j != k then
          candidates := candidates.push (← `(tactic| use $i:ident * $j:ident * $k:ident))
    for t in candidates do
      if ← closeWith do
        evalTactic t
        proverSearchDepth (budget - 1) (depth - 1) then
        return
  -- B6: simplifier over the whole context, then recurse
  if ← closeWith do
    evalTactic (← `(tactic| simp_all))
    proverSearchDepth (budget - 1) (depth - 1) then
    return
  throwError "prover_search: nothing closed at depth {depth}"

register_option prover_search.budget : Nat := {
  defValue := 1000
  descr := "node budget for `prover_search`"
}

/-- `prover_search` — bounded goal decomposition + backtracking.

Tries, in order, at every node: the hammer chain (`prover_finish`), case
split on local variables, induction on data variables, substitution along
equalities, witness search for existential goals, and `simp_all`; recursing
up to `depth` levels (default 3) with a node budget (default 1000, raised
via the `prover_search.budget` option). Deterministic and LLM-free. -/
elab "prover_search" n:(num)? : tactic => do
  let budget := prover_search.budget.get (← getOptions)
  let depth := match n with
    | some s => s.getNat
    | none => 3
  proverSearchDepth budget depth

end ProverSupport
