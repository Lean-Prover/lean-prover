import Mathlib

/-!
# Physics.Dimension — dimensional analysis for Lean

A minimal physical-dimension layer: a `Dimension` is a triple of integer
exponents over the base dimensions (Mass, Length, Time); a `Quantity` is a
real value tagged with a dimension. Multiplying/dividing quantities combines
their dimensions, so dimensionally-invalid equations are rejected by the type
system at compile time. This mirrors the dimensional-analysis formalizations
in the literature (e.g. "Formalizing Dimensional Analysis Using the Lean
Theorem Prover").

Mathlib ships no physics layer, so this is a self-contained addition.
-/

namespace Physics

/-- A physical dimension as exponents of Mass, Length, Time. -/
structure Dimension where
  M : ℤ
  L : ℤ
  T : ℤ
deriving DecidableEq

namespace Dimension

/-- Product of dimensions: exponents add. -/
def mul (a b : Dimension) : Dimension := ⟨a.M + b.M, a.L + b.L, a.T + b.T⟩

/-- Inverse of a dimension: exponents negate. -/
def inv (a : Dimension) : Dimension := ⟨-a.M, -a.L, -a.T⟩

/-- Power of a dimension: exponents scale. -/
def pow (a : Dimension) (n : ℕ) : Dimension := ⟨a.M * n, a.L * n, a.T * n⟩

instance : HMul Dimension Dimension Dimension := ⟨mul⟩
instance : HDiv Dimension Dimension Dimension := ⟨fun a b => a * b.inv⟩
instance : HPow Dimension ℕ Dimension := ⟨pow⟩

/-- Named base dimensions. -/
def mass : Dimension := ⟨1, 0, 0⟩
def length : Dimension := ⟨0, 1, 0⟩
def time : Dimension := ⟨0, 0, 1⟩

/-- Derived dimensions. -/
def velocity : Dimension := length / time
def acceleration : Dimension := velocity / time
def force : Dimension := mass * acceleration
def momentum : Dimension := mass * velocity
def energy : Dimension := mass * (velocity * velocity)
def power : Dimension := energy / time
def pressure : Dimension := force / (length * length)

end Dimension

/-- A real quantity tagged with a physical dimension. -/
structure Quantity (d : Dimension) where
  val : ℝ

namespace Quantity

/-- Multiply two quantities: the result carries the product dimension. -/
noncomputable def mul {d1 d2 : Dimension} (a : Quantity d1) (b : Quantity d2) : Quantity (d1 * d2) :=
  ⟨a.val * b.val⟩

/-- Divide two quantities: the result carries the quotient dimension. -/
noncomputable def div {d1 d2 : Dimension} (a : Quantity d1) (b : Quantity d2) : Quantity (d1 / d2) :=
  ⟨a.val / b.val⟩

/-- Raise a quantity to a power: the result carries the powered dimension. -/
noncomputable def pow {d : Dimension} (a : Quantity d) (n : ℕ) : Quantity (d ^ n) :=
  ⟨a.val ^ n⟩

/-- Scale a quantity by a dimensionless real: the dimension is unchanged. -/
noncomputable def smul {d : Dimension} (c : ℝ) (a : Quantity d) : Quantity d :=
  ⟨c * a.val⟩

/-- Named quantities. -/
noncomputable def mkQ {d : Dimension} (v : ℝ) : Quantity d := ⟨v⟩

end Quantity

open Dimension Quantity

-- Dimensional consistency theorems: each states that a composite quantity
-- has the expected dimension. These are integer-exponent equalities, closed
-- by `rfl`/`native_decide`.

/-- velocity = length / time (definitional). -/
theorem velocity_dim : velocity = length / time := rfl

/-- force = mass · acceleration. -/
theorem force_dim : force = mass * acceleration := rfl

/-- energy = mass · velocity². -/
theorem energy_dim : energy = mass * (velocity * velocity) := rfl

/-- kinetic energy has the energy dimension: mass · velocity² = energy. -/
theorem ke_is_energy : mass * (velocity ^ 2) = energy := by
  decide

/-- pressure = force / area. -/
theorem pressure_dim : pressure = force / (length * length) := rfl

/-- power = energy / time. -/
theorem power_dim : power = energy / time := rfl

end Physics
