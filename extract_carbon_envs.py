#!/usr/bin/env python3
"""
extract_carbon_envs.py – Extract all unique carbon environments from the nmrexp
SQLite database.

A carbon environment is characterised by:
  Local structure   – hybridisation, aromaticity, attached heteroatoms,
                      functional group, bonding pattern (graph neighbourhood)
  Extended context  – inductive / resonance effects up to `--radius` bonds,
                      ring membership, conjugation
  Stereochemistry   – whether the carbon is a stereocentre
  NMR context       – chemical-shift range observed in parent molecules
                      (verbose mode; NOTE: peaks are NOT individually assigned
                       to atoms – the range is approximate)

Usage
-----
  # Simple numbered list to stdout
  python extract_carbon_envs.py

  # Save simple list to a file
  python extract_carbon_envs.py -o environments.txt

  # Include full detail per environment
  python extract_carbon_envs.py --verbose

  # Verbose output saved to file
  python extract_carbon_envs.py --verbose -o environments_full.txt

  # Extend neighbourhood radius to 3 bonds (default: 2)
  python extract_carbon_envs.py --radius 3

  # Sort by occurrence count instead of structure type
  python extract_carbon_envs.py --sort count

  # Process only the first 500 molecules (quick test)
  python extract_carbon_envs.py --limit 500

  # Override the database path / table name on the command line
  python extract_carbon_envs.py --db /path/to/nmrexp.db --table my_table
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem.rdchem import HybridizationType
except ImportError:
    sys.exit(
        "RDKit is required.\n"
        "  conda : conda install -c conda-forge rdkit\n"
        "  pip   : pip install rdkit\n"
    )

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  – edit these two variables to match your setup
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH = "/home/vqire/Downloads/nmrexp_no_spectra.db"
TABLE_NAME = "nmr_data"

# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
#  Hybridisation labels
# ─────────────────────────────────────────────────────────────────────────────

HYBRID_LABEL: Dict = {
    HybridizationType.SP: "sp",
    HybridizationType.SP2: "sp²",
    HybridizationType.SP3: "sp³",
    HybridizationType.SP3D: "sp³d",
    HybridizationType.SP3D2: "sp³d²",
    HybridizationType.S: "s",
    HybridizationType.UNSPECIFIED: "?",
    HybridizationType.OTHER: "other",
}

# ─────────────────────────────────────────────────────────────────────────────
#  Functional-group SMARTS (priority order – first match wins).
#  The FIRST atom in each pattern is the target carbon.
# ─────────────────────────────────────────────────────────────────────────────

FG_SMARTS: List[Tuple[str, str]] = [
    # ── sp ───────────────────────────────────────────────────────────────────
    ("Nitrile C (C≡N)", "[CX2]#[NX1]"),
    ("Isonitrile C", "[CX1-]#[NX2+]"),
    ("Alkyne terminal C (≡CH)", "[CX2H1]#[CX2]"),
    ("Alkyne internal C (C≡C)", "[CX2]#[CX2]"),
    # ── sp²  C=O ─────────────────────────────────────────────────────────────
    ("Acyl halide C=O", "[CX3](=O)[F,Cl,Br,I]"),
    ("Anhydride C=O", "[CX3](=O)[OX2][CX3](=O)"),
    ("Carbonate C=O", "[OX2][CX3](=O)[OX2]"),
    ("Formate / Formyl C", "[CX3H1](=O)[OX2]"),
    ("Carboxylic acid C=O", "[CX3](=O)[OX2H1]"),
    ("Ester / Lactone C=O", "[CX3;H0](=O)[OX2][!H]"),
    ("Thioester C=O", "[CX3](=O)[SX2]"),
    ("Urea / Carbamate C=O", "[NX3][CX3](=O)[NX3,OX2]"),
    ("Amide / Lactam C=O", "[CX3](=O)[NX3]"),
    ("Aldehyde C=O", "[CX3H1]=O"),
    ("Ketone C=O", "[CX3;H0](=O)([#6])[#6]"),
    # ── sp²  C=N / C=C ───────────────────────────────────────────────────────
    ("Oxime C=N", "[CX3]=[NX2][OX2H]"),
    ("Imine C=N", "[CX3]=[NX2]"),
    ("Enol ether vinyl C", "[CX3]([OX2H0][#6])=[CX3]"),
    ("Enamide / Enamine vinyl C", "[CX3]([NX3])=[CX3]"),
    ("Vinyl halide C", "[CX3](=[CX3])[F,Cl,Br,I]"),
    ("Alkene C=C", "[CX3]=[CX3]"),
    # ── sp²  aromatic ────────────────────────────────────────────────────────
    ("Aromatic CH", "[cH]"),
    ("Aromatic C–heteroatom (direct)", "[c;H0][O,N,S,F,Cl,Br,I,Si,P]"),
    ("Aromatic quaternary C", "[c;H0]"),
    # ── sp³  strained rings ───────────────────────────────────────────────────
    ("Epoxide C", "[CX4]1[OX2][CX4]1"),
    ("Cyclopropane C", "[CX4]1[CX4][CX4]1"),
    ("Aziridine C", "[CX4]1[NX3][CX4]1"),
    # ── sp³  alpha-C to heteroatom (ordered by electron-withdrawal) ──────────
    ("Alpha-C to acyl halide", "[CX4][CX3](=O)[F,Cl,Br,I]"),
    ("Alpha-C to carbonyl (sp³)", "[CX4][CX3]=O"),
    ("Benzylic C (sp³)", "[CX4][c]"),
    ("Propargylic C (sp³)", "[CX4][CX2]#[CX2]"),
    ("Allylic C (sp³)", "[CX4][CX3]=[CX3]"),
    ("Hemiacetal / Acetal C", "[CX4;H1,H0]([OX2H])[OX2]"),
    ("Alpha-C to ether O (sp³)", "[CX4][OX2][!H]"),
    ("Alpha-C to hydroxyl O (sp³)", "[CX4][OX2H]"),
    ("Alpha-C to amide N (sp³)", "[CX4][NX3][CX3]=O"),
    ("Alpha-C to amine N (sp³)", "[CX4][NX3;!$(N=*)]"),
    ("Alpha-C to F", "[CX4][F]"),
    ("Alpha-C to Cl / Br / I", "[CX4][Cl,Br,I]"),
    ("Alpha-C to sulfoxide S (sp³)", "[CX4][SX3]=O"),
    ("Alpha-C to sulfone S (sp³)", "[CX4][SX4](=O)=O"),
    ("Alpha-C to thioether S (sp³)", "[CX4][SX2][!H]"),
    ("Alpha-C to phosphorus (sp³)", "[CX4][PX4,PX3]"),
    ("Alpha-C to silicon (sp³)", "[CX4][Si]"),
    # ── sp³  general ─────────────────────────────────────────────────────────
    ("Methyl C (sp³)", "[CX4H3]"),
    ("Methylene C (sp³)", "[CX4H2]"),
    ("Methine C (sp³)", "[CX4H1]"),
    ("Quaternary sp³ C", "[CX4H0]"),
]

# Pre-compile all SMARTS once at import time
_FG_COMPILED: List[Tuple[str, Optional[Chem.Mol]]] = [
    (name, Chem.MolFromSmarts(sma)) for name, sma in FG_SMARTS
]


# ─────────────────────────────────────────────────────────────────────────────
#  RDKit helper functions
# ─────────────────────────────────────────────────────────────────────────────


def detect_fg(mol: Chem.Mol, atom_idx: int) -> str:
    """Return the highest-priority functional-group label for this carbon.

    Uses uniquify=False so that symmetric patterns (e.g. [CX3]=[CX3]) yield
    both orderings and the target atom is reliably found as match[0].
    """
    for name, pat in _FG_COMPILED:
        if pat is None:
            continue
        for match in mol.GetSubstructMatches(pat, uniquify=False):
            if match[0] == atom_idx:
                return name
    return "Unclassified C"


def get_env_smiles(mol: Chem.Mol, atom_idx: int, radius: int) -> str:
    """
    Canonical SMILES fragment centred on `atom_idx` covering all atoms within
    `radius` bonds.  Used as the deduplication key for environments.

    Notes
    -----
    FindAtomEnvironmentOfRadiusN returns ALL bonds up to `radius` when atoms
    exist at that distance, but returns an empty collection when `radius`
    exceeds the molecule's extent from that atom.  We walk down from `radius`
    to 1 to find the largest available environment.
    """
    env_bonds: List[int] = []
    for r in range(radius, 0, -1):
        bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx, useHs=False))
        if bonds:
            env_bonds = bonds
            break

    if not env_bonds:
        # Isolated atom (e.g. a single-carbon molecule) – return element symbol
        return mol.GetAtomWithIdx(atom_idx).GetSymbol()

    atoms_in_env: set = set()
    for bid in env_bonds:
        b = mol.GetBondWithIdx(bid)
        atoms_in_env.add(b.GetBeginAtomIdx())
        atoms_in_env.add(b.GetEndAtomIdx())

    try:
        smi = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=list(atoms_in_env),
            bondsToUse=env_bonds,
            rootedAtAtom=atom_idx,
            canonical=True,
        )
        return smi if smi else mol.GetAtomWithIdx(atom_idx).GetSymbol()
    except Exception:
        return mol.GetAtomWithIdx(atom_idx).GetSymbol()


def hetero_neighbors(mol: Chem.Mol, atom_idx: int) -> List[str]:
    """Sorted list of non-C / non-H element symbols directly bonded to this atom."""
    seen: set = set()
    for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        sym = nbr.GetSymbol()
        if sym not in ("C", "H"):
            seen.add(sym)
    return sorted(seen)


def distant_heteroatoms(
    mol: Chem.Mol, atom_idx: int, min_dist: int = 2, max_dist: int = 3
) -> List[str]:
    """
    Heteroatom element symbols at `min_dist`–`max_dist` bonds from this carbon
    (inductive-effect range).
    """
    seen: set = set()
    visited = {atom_idx}
    current = {atom_idx}
    for dist in range(1, max_dist + 1):
        nxt: set = set()
        for ai in current:
            for nbr in mol.GetAtomWithIdx(ai).GetNeighbors():
                ni = nbr.GetIdx()
                if ni not in visited:
                    visited.add(ni)
                    nxt.add(ni)
                    if dist >= min_dist:
                        sym = nbr.GetSymbol()
                        if sym not in ("C", "H"):
                            seen.add(sym)
        current = nxt
    return sorted(seen)


def smallest_ring_size(mol: Chem.Mol, atom_idx: int) -> Optional[int]:
    sizes = [len(r) for r in mol.GetRingInfo().AtomRings() if atom_idx in r]
    return min(sizes) if sizes else None


def is_conjugated(mol: Chem.Mol, atom_idx: int) -> bool:
    return any(b.GetIsConjugated() for b in mol.GetAtomWithIdx(atom_idx).GetBonds())


# ─────────────────────────────────────────────────────────────────────────────
#  Data model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CarbonEnv:
    # ── Identity ──────────────────────────────────────────────────────────────
    env_smiles: str  # canonical SMILES fragment (key)
    hybridization: str  # sp / sp² / sp³ …
    is_aromatic: bool
    functional_group: str  # highest-priority FG label
    heteroatoms_d1: List[str]  # heteroatoms at distance 1 (direct bonds)
    heteroatoms_d23: List[str]  # heteroatoms at distance 2–3 (inductive range)
    is_chiral: bool  # stereocentre in at least one parent molecule
    in_ring: bool
    ring_size: Optional[int]  # smallest ring containing this atom
    conjugated: bool  # any adjacent conjugated bond
    atom_count: int = 0  # number of atoms in the environment

    # ── Statistics ────────────────────────────────────────────────────────────
    count: int = 0
    example_ids: List[int] = field(default_factory=list)  # up to 5 IDs

    # ── Approximate NMR context ───────────────────────────────────────────────
    # All ¹³C peaks from every parent molecule that contains this environment.
    # These are NOT assigned to this specific carbon; treat as rough indicator.
    parent_ppm: List[float] = field(default_factory=list)

    @property
    def ppm_stats(self) -> str:
        if not self.parent_ppm:
            return "n/a"
        lo = min(self.parent_ppm)
        hi = max(self.parent_ppm)
        avg = sum(self.parent_ppm) / len(self.parent_ppm)
        return f"{lo:.1f} – {hi:.1f} ppm  (mean {avg:.1f}, n={len(self.parent_ppm)})"


# ─────────────────────────────────────────────────────────────────────────────
#  Extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract(
    db_path: str = DB_PATH,
    table: str = TABLE_NAME,
    radius: int = 2,
    limit: Optional[int] = None,
) -> Dict[str, CarbonEnv]:
    """
    Query the database and return a dict mapping env_smiles → CarbonEnv.
    Every carbon atom in every molecule is processed; identical environments
    (same canonical neighbourhood SMILES) are merged.
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    q = f'SELECT id, smiles, peaks FROM "{table}"'
    if limit:
        q += f" LIMIT {limit}"

    rows = cur.execute(q).fetchall()
    con.close()

    envs: Dict[str, CarbonEnv] = {}
    processed = 0
    parse_errs = 0

    for row_id, smiles, peaks_json in rows:
        if not smiles:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            parse_errs += 1
            continue
        processed += 1

        # Collect all ¹³C ppm values for this molecule (not per-atom assigned)
        mol_ppm: List[float] = []
        if peaks_json:
            try:
                for p in json.loads(peaks_json):
                    v = p.get("ppm")
                    if isinstance(v, (int, float)):
                        mol_ppm.append(float(v))
            except Exception:
                pass

        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                continue
            idx = atom.GetIdx()

            key = get_env_smiles(mol, idx, radius)

            # Compute atom count for this environment
            env_bonds = []
            for r in range(radius, 0, -1):
                bonds = list(
                    Chem.FindAtomEnvironmentOfRadiusN(mol, r, idx, useHs=False)
                )
                if bonds:
                    env_bonds = bonds
                    break
            atoms_in_env = set()
            for bid in env_bonds:
                b = mol.GetBondWithIdx(bid)
                atoms_in_env.add(b.GetBeginAtomIdx())
                atoms_in_env.add(b.GetEndAtomIdx())
            if not atoms_in_env:
                atoms_in_env.add(idx)

            if key not in envs:
                envs[key] = CarbonEnv(
                    env_smiles=key,
                    hybridization=HYBRID_LABEL.get(atom.GetHybridization(), "?"),
                    is_aromatic=atom.GetIsAromatic(),
                    functional_group=detect_fg(mol, idx),
                    heteroatoms_d1=hetero_neighbors(mol, idx),
                    heteroatoms_d23=distant_heteroatoms(mol, idx, 2, 3),
                    is_chiral=(
                        atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED
                    ),
                    in_ring=atom.IsInRing(),
                    ring_size=smallest_ring_size(mol, idx),
                    conjugated=is_conjugated(mol, idx),
                    atom_count=len(atoms_in_env),
                )

            env = envs[key]
            env.count += 1
            if row_id not in env.example_ids and len(env.example_ids) < 5:
                env.example_ids.append(row_id)
            env.parent_ppm.extend(mol_ppm)

        if processed % 1000 == 0:
            print(f"    … {processed:,} molecules processed", file=sys.stderr)

    print(
        f"[*] Finished: {processed:,} molecules processed, "
        f"{parse_errs} parse errors.",
        file=sys.stderr,
    )
    return envs


# ─────────────────────────────────────────────────────────────────────────────
#  Sorting
# ─────────────────────────────────────────────────────────────────────────────

_HYBRID_ORDER = {"sp": 0, "sp²": 1, "sp³": 2}


def _sort_key(env: CarbonEnv, by: str):
    if by == "count":
        return (-env.count, env.hybridization, env.functional_group)
    # default: by structure (sp → sp² → sp³, aromatic first, then FG name)
    return (
        _HYBRID_ORDER.get(env.hybridization, 9),
        not env.is_aromatic,
        env.functional_group,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Output formatters
# ─────────────────────────────────────────────────────────────────────────────


def _file_header(
    db_path: str, table: str, radius: int, total: int, verbose: bool = False
) -> List[str]:
    suffix = "  (verbose)" if verbose else ""
    return [
        f"UNIQUE CARBON ENVIRONMENTS{suffix}",
        "=" * 80,
        f"Database         : {db_path}",
        f"Table            : {table}",
        f"Neighbourhood    : {radius} bonds",
        f"Total environments: {total}",
        "",
    ]


def format_simple(
    envs: Dict[str, CarbonEnv],
    db_path: str,
    table: str,
    radius: int,
    sort_by: str = "structure",
) -> str:
    """
    Compact numbered table.  Columns:
      No. | Hybr. | Arom. | Functional Group | Count | Environment SMILES
    """
    items = sorted(envs.values(), key=lambda e: _sort_key(e, sort_by))
    lines = _file_header(db_path, table, radius, len(items))

    W_HYBD = 6
    W_AROM = 6
    W_FG = 40
    W_CNT = 8

    header = (
        f"{'No.':<5} {'Hybr.':<{W_HYBD}} {'Arom.':<{W_AROM}} "
        f"{'Functional Group':<{W_FG}} {'Count':<{W_CNT}} Atoms  Environment SMILES"
    )
    lines.append(header)
    lines.append("─" * 140)

    for i, env in enumerate(items, 1):
        arom = "Yes" if env.is_aromatic else "No"
        lines.append(
            f"{i:<5} {env.hybridization:<{W_HYBD}} {arom:<{W_AROM}} "
            f"{env.functional_group:<{W_FG}} {env.count:<{W_CNT}} {env.atom_count:<6} {env.env_smiles}"
        )

    return "\n".join(lines)


def format_verbose(
    envs: Dict[str, CarbonEnv],
    db_path: str,
    table: str,
    radius: int,
    sort_by: str = "structure",
) -> str:
    """
    Full detail block for every environment.
    """
    items = sorted(envs.values(), key=lambda e: _sort_key(e, sort_by))
    lines = _file_header(db_path, table, radius, len(items), verbose=True)

    for i, env in enumerate(items, 1):
        arom_s = "Yes" if env.is_aromatic else "No"
        chiral_s = "Yes" if env.is_chiral else "No"
        conj_s = "Yes" if env.conjugated else "No"
        ring_s = (
            f"Yes – smallest ring: {env.ring_size}-membered" if env.in_ring else "No"
        )
        d1_s = ", ".join(env.heteroatoms_d1) or "none"
        d23_s = ", ".join(env.heteroatoms_d23) or "none"
        ids_s = ", ".join(map(str, env.example_ids)) or "—"

        lines += [
            "─" * 80,
            f"[{i:04d}]  {env.functional_group}",
            f"",
            f"  Local chemical structure",
            f"  ├─ Hybridisation      : {env.hybridization}",
            f"  ├─ Aromatic           : {arom_s}",
            f"  ├─ Heteroatoms (d=1)  : {d1_s}",
            f"  ├─ Functional group   : {env.functional_group}",
            f"  ├─ Atoms in env       : {env.atom_count}",
            f"  └─ Env. SMILES (r={radius})  : {env.env_smiles}",
            f"",
            f"  Extended chemical context",
            f"  ├─ Heteroatoms (d=2–3): {d23_s}",
            f"  ├─ Conjugated system  : {conj_s}",
            f"  └─ In ring            : {ring_s}",
            f"",
            f"  Stereochemistry",
            f"  └─ Stereocentre       : {chiral_s}",
            f"",
            f"  Database statistics",
            f"  ├─ Occurrences        : {env.count}",
            f"  └─ Example mol IDs    : {ids_s}",
            f"",
            f"  Approximate NMR context  (¹³C, not per-atom assigned)",
            f"  └─ Parent-mol peak range: {env.ppm_stats}",
            f"     ↳ All peaks from all parent molecules containing this",
            f"       environment are pooled; they are NOT assigned to this",
            f"       specific carbon.  Use as a rough chemical-shift indicator.",
            f"",
        ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="extract_carbon_envs.py",
        description=(
            "Extract all unique carbon environments from the nmrexp SQLite database."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Write output to FILE instead of stdout",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Include full structural and NMR detail per environment",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=int,
        default=2,
        metavar="N",
        help="Bond-radius for neighbourhood SMILES fragment  [default: 2]",
    )
    parser.add_argument(
        "--sort",
        choices=["structure", "count"],
        default="structure",
        metavar="[structure|count]",
        help="Sort order: 'structure' (sp→sp²→sp³, default) or 'count' (most common first)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N molecules  (useful for quick tests)",
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        metavar="PATH",
        help=f"Path to the SQLite database  [default: {DB_PATH}]",
    )
    parser.add_argument(
        "--table",
        default=TABLE_NAME,
        metavar="NAME",
        help=f"Table name inside the database  [default: {TABLE_NAME}]",
    )

    args = parser.parse_args()

    print(f"[*] Database  : {args.db}", file=sys.stderr)
    print(f"[*] Table     : {args.table}", file=sys.stderr)
    print(f"[*] Radius    : {args.radius} bonds", file=sys.stderr)
    if args.limit:
        print(f"[*] Limit     : {args.limit} molecules", file=sys.stderr)

    envs = extract(
        db_path=args.db,
        table=args.table,
        radius=args.radius,
        limit=args.limit,
    )
    print(f"[*] Unique environments found: {len(envs)}", file=sys.stderr)

    formatter = format_verbose if args.verbose else format_simple
    text = formatter(envs, args.db, args.table, args.radius, sort_by=args.sort)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"[*] Written to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
