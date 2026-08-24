import Mathlib
import ProverSupport

open BigOperators Nat Finset

theorem t2 (a b : ℕ) : a + b = b + a := by
  prover_finish
