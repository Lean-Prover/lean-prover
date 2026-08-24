"""leagent lean_extract wrapper: version gating + corpus row conversion."""

from __future__ import annotations

import json

import pytest

from benchmark import mine_corpus as mc


def test_library_path_for_toolchain() -> None:
    p = mc.library_path_for("v4.32.2")
    assert p.parts[-2:] == ("lib", "lean") and "leanprover--lean4---v4.32.2" in str(p)


def test_extract_refuses_version_mismatch(tmp_path, monkeypatch) -> None:
    lean_dir = tmp_path / "lean"
    lean_dir.mkdir()
    (lean_dir / "lean-toolchain").write_text("leanprover/lean4:v4.33.0\n")
    fake_bin = tmp_path / "lean_extract-v4.32.2"
    fake_bin.write_text("")
    monkeypatch.setattr(mc, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="targets Lean 4.32.2 but project pins v4.33.0"):
        mc.extract(fake_bin, ["X"], tmp_path / "out", lean_dir=lean_dir)
    # missing toolchain lib is caught too
    ok_bin = tmp_path / "lean_extract-v4.33.0"
    ok_bin.write_text("")
    monkeypatch.setattr(mc, "library_path_for",
                        lambda _v: tmp_path / "no-such-toolchain" / "lib" / "lean")
    with pytest.raises(SystemExit, match="not installed"):
        mc.extract(ok_bin, ["X"], tmp_path / "out", lean_dir=lean_dir)


def test_mined_to_rows_filters_and_shapes(tmp_path) -> None:
    f = tmp_path / "x.jsonl"
    rows = [
        {"name": "good", "kind": "theorem", "signature": "theorem good (n : ℕ) : n + 0 = n",
         "proof": "simp"},
        {"name": "native", "kind": "theorem", "signature": "theorem native : 1 = 1",
         "proof": "prover_finish"},  # native tag → dropped
        {"name": "a_def", "kind": "def", "signature": "def a := 1", "proof": "x"},
        {"no_proof": True, "kind": "theorem", "signature": "theorem no_proof : True"},
    ][:3] + [{"name": "no_sig", "kind": "theorem", "proof": "simp"}]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    out = mc.mined_to_rows(f)
    assert len(out) == 1
    r = out[0]
    assert r["id"] == "mined_good"
    assert r["statement"].startswith("theorem good (n : ℕ) : n + 0 = n := by\nsimp")


def test_append_to_corpus_idempotent(tmp_path) -> None:
    c = tmp_path / "corpus.jsonl"
    row = {"id": "m1", "difficulty": "mined", "statement": "s"}
    assert mc.append_to_corpus([row], c) == 1
    assert mc.append_to_corpus([row], c) == 0
    assert len(c.read_text().splitlines()) == 1
