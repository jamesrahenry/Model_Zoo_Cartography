"""Pin null_baseline/refit_trained.py's population protocol (§3.6 / §4.7).

After the 2026-08-18 correction, the paper states the protocol precisely:
the validation population gates polish acceptance (so its numbers are
validation, not untouched held-out), and the transfer population is reserved
untouched for the out-of-population read. Nothing enforced that separation
until now. These tests pin the declared populations; if a family is ever
moved between roles, the paper's §4.7 and FINDINGS F4 must move with it.

Not covered here: that main()'s selection loop never *evaluates* transfer —
that is a data-dependent code path (needs corpus weights); the structural
disjointness below is the cheap invariant that makes the claim auditable.
"""
from __future__ import annotations

import itertools

from refit_trained import POPULATIONS


def test_roles_are_pairwise_disjoint_within_every_population():
    for name, pop in POPULATIONS.items():
        for a, b in itertools.combinations(("fit", "val", "transfer"), 2):
            shared = set(pop[a]) & set(pop[b])
            assert not shared, f"{name}: {a} and {b} share {shared}"


def test_every_population_declares_all_three_roles_nonempty():
    for name, pop in POPULATIONS.items():
        for role in ("fit", "val", "transfer"):
            assert pop[role], f"{name} has an empty {role} set"


def test_documented_protocol_facts():
    """The specific facts §4.7 and FINDINGS F4 state about the populations."""
    conv = POPULATIONS["converged"]
    # The untouched out-of-population read is the C=50 partial learners.
    assert conv["transfer"] == ["sweep_c50_head"]
    # Validation is the two C=10 probe families in both regimes — the same
    # set that gates polish acceptance (hence: validation, not held-out).
    assert set(conv["val"]) == {"probe_d32_c10", "probe_d32_c10_head"}
    assert set(POPULATIONS["mixed"]["val"]) == set(conv["val"])
    # The mixed (M2) regime moves partial learners INTO fit deliberately —
    # its transfer set must then be different families, not the ones it fit.
    mixed = POPULATIONS["mixed"]
    assert "sweep_c50_head" in mixed["fit"]
    assert "sweep_c50_head" not in mixed["transfer"]
