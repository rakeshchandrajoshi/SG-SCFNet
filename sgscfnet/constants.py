from __future__ import annotations

FRESHNESS = ("Highly Fresh", "Fresh", "Not Fresh")
FRESHNESS_TO_ID = {name.casefold(): i for i, name in enumerate(FRESHNESS)}
FFE_SPECIES = (
    "Chanos Chanos",
    "Eleutheronema Tetradactylum",
    "Johnius Trachycephalus",
    "Nibea Albiflora",
    "Oreochromis Mossambicus",
    "Oreochromis Niloticus",
    "Rastrelliger Faughni",
    "Upeneus Moluccensis",
)
# Rows are true freshness classes; columns are predicted freshness classes.
SAFETY_COST = ((0.0, 1.0, 2.0), (1.0, 0.0, 1.0), (5.0, 3.0, 0.0))
