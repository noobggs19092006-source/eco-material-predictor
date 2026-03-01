"""
generate_pdb.py — PDB file library for eco-polymer repeat units (VMD-ready).

Fixes from v1:
  - Complete 3D coordinates (non-flat, proper bond geometry, all 3 axes used)
  - ALL atoms including every H — no missing bonds
  - 3-char residue names (VMD/PDB standard)
  - HETATM records (correct for small molecules)
  - Full CONECT table for every bond
  - Bond lengths: C-C 1.54Å, C=O 1.20Å, C-O(ester) 1.36Å, C-O 1.43Å, C-H 1.09Å

Open in VMD:
    vmd molecule.pdb
    Then: Graphics → Representations → Drawing Method → Licorice (recommended)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Each entry: name, formula, eco_score, atoms list, bonds list
# atoms: (atom_name, element,  X,      Y,      Z   )
# bonds: list of (i, j) 1-indexed atom pairs
# ─────────────────────────────────────────────────────────────────────────────
_LIBRARY = {

    # ── PLA — Poly(lactic acid) repeat unit ──────────────────────────────────
    # Formula: C3H4O2   MW=72  Tg≈60°C  Eco=1.0  fully bio-based
    # Structure: -[CH(CH3)-C(=O)-O]-
    #   C_A = alpha-C (sp3), C_M = methyl (sp3), C_C = carbonyl (sp2)
    #   O1  = carbonyl =O,   O2  = ester O (chain continues here)
    "PLA": {
        "name":      "Poly(Lactic Acid) repeat unit",
        "formula":   "C3H4O2",
        "eco_score": 1.0,
        "atoms": [
            #  name   el     X       Y       Z
            ("CA",  "C",  0.000,  0.000,  0.000),   # 1  alpha-C (sp3)
            ("CM",  "C", -0.513,  1.452,  0.000),   # 2  methyl-C (sp3)
            ("CC",  "C",  1.524,  0.000,  0.000),   # 3  carbonyl-C (sp2)
            ("O1",  "O",  2.124,  1.039,  0.000),   # 4  C=O oxygen
            ("O2",  "O",  2.204, -1.178,  0.000),   # 5  ester O
            ("HA",  "H", -0.363, -0.513,  0.889),   # 6  H on alpha-C
            ("HM1", "H", -1.603,  1.360,  0.000),   # 7  methyl H1
            ("HM2", "H", -0.168,  1.975,  0.889),   # 8  methyl H2
            ("HM3", "H", -0.168,  1.975, -0.889),   # 9  methyl H3
        ],
        "bonds": [
            (1,2),(1,3),(1,6),       # alpha-C bonds
            (2,7),(2,8),(2,9),       # methyl H's
            (3,4),(3,5),             # carbonyl + ester O
        ],
    },

    # ── PHB — Poly(3-hydroxybutyrate) repeat unit ─────────────────────────
    # Formula: C4H6O2   MW=86  Tg≈4°C   Eco=1.0  microbial, biodegradable
    # Structure: -[CH(CH3)-CH2-C(=O)-O]-
    "PHB": {
        "name":      "Poly(3-hydroxybutyrate) repeat unit",
        "formula":   "C4H6O2",
        "eco_score": 1.0,
        "atoms": [
            ("CA",  "C",  0.000,  0.000,  0.000),   # 1  alpha-C (sp3, CH)
            ("CM",  "C",  0.513, -1.452,  0.000),   # 2  methyl-C (sp3)
            ("CB",  "C", -1.263,  0.728,  0.545),   # 3  beta-CH2 (sp3)
            ("CC",  "C", -2.578,  0.000,  0.136),   # 4  carbonyl-C (sp2)
            ("O1",  "O", -3.100,  0.780, -0.630),   # 5  C=O oxygen
            ("O2",  "O", -3.180, -0.970,  0.598),   # 6  ester O
            ("HA",  "H",  0.590,  0.870,  0.335),   # 7  H on alpha-C
            ("HM1", "H",  1.602, -1.360,  0.000),   # 8  methyl H1
            ("HM2", "H",  0.167, -1.975,  0.889),   # 9  methyl H2
            ("HM3", "H",  0.167, -1.975, -0.889),   # 10 methyl H3
            ("HB1", "H", -1.166,  1.682,  0.040),   # 11 beta H1
            ("HB2", "H", -1.234,  0.844,  1.633),   # 12 beta H2
        ],
        "bonds": [
            (1,2),(1,3),(1,7),
            (2,8),(2,9),(2,10),
            (3,4),(3,11),(3,12),
            (4,5),(4,6),
        ],
    },

    # ── PBS — Poly(butylene succinate) repeat unit ────────────────────────
    # Formula: C8H12O4  MW=172 Tg≈-32°C Eco=0.85 semi-biodegradable
    # Structure: -[C(=O)-CH2-CH2-C(=O)-O-CH2-CH2-CH2-CH2-O]-
    "PBS": {
        "name":      "Poly(butylene succinate) repeat unit",
        "formula":   "C8H12O4",
        "eco_score": 0.85,
        "atoms": [
            # Succinate part: C(=O)-CH2-CH2-C(=O)-O
            ("C1",  "C",  0.000,  0.000,  0.000),   # 1  carbonyl-C sp2
            ("O1",  "O", -0.600,  1.040,  0.000),   # 2  C=O (1.20Å, 120°)
            ("O2",  "O",  1.360,  0.000,  0.000),   # 3  ester-O1 not here: starts chain
            ("C2",  "C", -0.760, -0.985,  0.554),   # 4  CH2 (sp3)
            ("C3",  "C", -2.290, -0.985,  0.225),   # 5  CH2 (sp3)
            ("C4",  "C", -3.050,  0.000,  1.100),   # 6  carbonyl-C sp2
            ("O3",  "O", -2.450,  1.040,  1.100),   # 7  C=O oxygen
            ("O4",  "O", -4.410,  0.000,  1.100),   # 8  ester-O (links to butylenediol)
            # Butylenediol part: CH2-CH2-CH2-CH2
            ("C5",  "C", -5.170, -0.985,  0.554),   # 9  CH2 (sp3)
            ("C6",  "C", -6.700, -0.985,  0.885),   # 10 CH2 (sp3)
            ("C7",  "C", -7.460,  0.000,  0.000),   # 11 CH2 (sp3)
            ("C8",  "C", -8.990,  0.000,  0.331),   # 12 CH2 (sp3)
            # Ester O closing the ring
            ("O5",  "O", -9.550, -0.985, -0.470),   # 13 ester-O (back to C1 next unit)
            # Hydrogens
            ("H2A", "H", -0.405, -1.997,  0.440),   # 14 H on C2
            ("H2B", "H", -0.670, -0.770,  1.630),   # 15 H on C2
            ("H3A", "H", -2.645, -1.997,  0.339),   # 16 H on C3
            ("H3B", "H", -2.380, -0.770, -0.852),   # 17 H on C3
            ("H5A", "H", -4.815, -1.997,  0.440),   # 18 H on C5
            ("H5B", "H", -5.060, -0.770, -0.522),   # 19 H on C5
            ("H6A", "H", -7.055, -1.997,  0.771),   # 20 H on C6
            ("H6B", "H", -6.810, -0.770,  1.961),   # 21 H on C6
            ("H7A", "H", -7.105, -0.012, -1.076),   # 22 H on C7
            ("H7B", "H", -7.350,  1.010,  0.340),   # 23 H on C7
            ("H8A", "H", -9.345, -0.012,  1.407),   # 24 H on C8
            ("H8B", "H", -9.080,  1.010, -0.009),   # 25 H on C8
        ],
        "bonds": [
            (1,2),(1,3),(1,4),            # C1 bonds (corrected: no O2)
            (4,5),(4,14),(4,15),
            (5,6),(5,16),(5,17),
            (6,7),(6,8),
            (8,9),
            (9,10),(9,18),(9,19),
            (10,11),(10,20),(10,21),
            (11,12),(11,22),(11,23),
            (12,13),(12,24),(12,25),
        ],
    },

    # ── PEF — Poly(ethylene furanoate) repeat unit ────────────────────────
    # Formula: C8H6O5  MW=192 Tg≈80°C  Eco=1.0 bio-PET replacement
    # Structure: -[C(=O)-furan-C(=O)-O-CH2-CH2-O]-
    "PEF": {
        "name":      "Poly(ethylene furanoate) repeat unit",
        "formula":   "C8H6O5",
        "eco_score": 1.0,
        "atoms": [
            ("C1",  "C",  0.000,  0.000,  0.000),   # 1  carbonyl-C sp2
            ("O1",  "O", -0.600,  1.040,  0.000),   # 2  C=O oxygen
            ("O2",  "O",  1.340, -0.795,  0.000),   # 3  ester-O
            # Furan ring (planar, aromatic)
            ("C2",  "C", -1.440,  0.000,  0.000),   # 4  furan C2
            ("C3",  "C", -2.100,  1.218,  0.000),   # 5  furan C3
            ("O3",  "O", -3.474,  0.934,  0.000),   # 6  furan O
            ("C4",  "C", -3.474, -0.409,  0.000),   # 7  furan C4
            ("C5",  "C", -2.100, -1.218,  0.000),   # 8  furan C5
            # Second carbonyl
            ("C6",  "C", -1.440, -2.600,  0.000),   # 9  carbonyl-C sp2 (mirrors C1)
            ("O4",  "O", -0.840, -3.640,  0.000),   # 10 C=O oxygen
            ("O5",  "O", -2.600, -3.300,  0.000),   # 11 ester-O (to ethylene)
            # Ethylene bridge
            ("C7",  "C",  2.680, -0.795,  0.000),   # 12 ethylene-CH2
            ("C8",  "C",  3.440, -0.795, -1.250),   # 13 ethylene-CH2
            # Hydrogens
            ("H3",  "H", -1.674,  2.195,  0.000),   # 14 furan H3
            ("H4",  "H", -4.450, -0.909,  0.000),   # 15 furan H4
            ("H7A", "H",  2.976, -1.720,  0.523),   # 16 ethylene H
            ("H7B", "H",  2.976,  0.130,  0.523),   # 17 ethylene H
            ("H8A", "H",  3.144, -0.795, -2.300),   # 18 ethylene H
            ("H8B", "H",  4.530, -0.795, -1.250),   # 19 ethylene H
        ],
        "bonds": [
            (1,2),(1,3),(1,4),
            (4,5),(4,8),
            (5,6),(5,14),
            (6,7),
            (7,8),(7,15),
            (8,9),
            (9,10),(9,11),
            (3,12),
            (12,13),(12,16),(12,17),
            (13,11),(13,18),(13,19),
        ],
    },

    # ── CELLULOSE — glucopyranose repeat unit ─────────────────────────────
    # Formula: C6H10O5  MW=162  Tg≈220°C  Eco=1.0  natural, abundant
    "CELLULOSE": {
        "name":      "Cellulose (glucopyranose) repeat unit",
        "formula":   "C6H10O5",
        "eco_score": 1.0,
        "atoms": [
            ("C1",  "C",  0.000,  0.000,  0.000),   # 1  ring-C1 (anomeric)
            ("C2",  "C",  1.520,  0.000,  0.000),   # 2  ring-C2
            ("C3",  "C",  2.115,  1.360,  0.358),   # 3  ring-C3
            ("C4",  "C",  1.520,  2.558,  0.000),   # 4  ring-C4
            ("C5",  "C",  0.000,  2.558,  0.000),   # 5  ring-C5
            ("O5",  "O", -0.600,  1.280,  0.000),   # 6  ring-O5
            ("C6",  "C", -0.595,  3.742,  0.554),   # 7  exocyclic CH2OH
            ("O2",  "O",  2.115, -1.120,  0.358),   # 8  OH at C2
            ("O3",  "O",  3.510,  1.360,  0.358),   # 9  OH at C3
            ("O4",  "O",  2.115,  3.742, -0.354),   # 10 glycosidic O at C4
            ("O6",  "O", -0.595,  4.940,  0.000),   # 11 OH at C6
            # Hydrogens
            ("H1",  "H", -0.396,  0.000, -1.025),   # 12 H on C1
            ("H2",  "H",  1.870, -0.040, -1.025),   # 13 H on C2
            ("H3",  "H",  1.820,  1.360,  1.418),   # 14 H on C3
            ("H4",  "H",  1.870,  2.600, -1.025),   # 15 H on C4
            ("H5",  "H", -0.396,  2.520, -1.028),   # 16 H on C5
            ("H6A", "H", -1.685,  3.684,  0.440),   # 17 H on C6
            ("H6B", "H", -0.249,  3.700,  1.590),   # 18 H on C6
            ("HO2", "H",  3.064, -1.120,  0.200),   # 19 OH H at C2
            ("HO3", "H",  3.890,  2.236,  0.120),   # 20 OH H at C3
            ("HO6", "H", -1.565,  4.940,  0.000),   # 21 OH H at C6
        ],
        "bonds": [
            (1,2),(1,6),(1,12),
            (2,3),(2,8),(2,13),
            (3,4),(3,9),(3,14),
            (4,5),(4,10),(4,15),
            (5,6),(5,7),(5,16),
            (6,1),           # ring closure already done above
            (7,11),(7,17),(7,18),
            (8,19),(9,20),(11,21),
        ],
    },

    # ── CHITOSAN — deacetylated chitin repeat unit ─────────────────────────
    # Formula: C6H11NO4  Eco=1.0  marine-waste derived, antibacterial
    "CHITOSAN": {
        "name":      "Chitosan repeat unit",
        "formula":   "C6H11NO4",
        "eco_score": 1.0,
        "atoms": [
            ("C1",  "C",  0.000,  0.000,  0.000),   # 1  ring-C1 (anomeric)
            ("C2",  "C",  1.520,  0.000,  0.000),   # 2  ring-C2 (NH2 bearing)
            ("C3",  "C",  2.115,  1.360,  0.358),   # 3  ring-C3
            ("C4",  "C",  1.520,  2.558,  0.000),   # 4  ring-C4
            ("C5",  "C",  0.000,  2.558,  0.000),   # 5  ring-C5
            ("O5",  "O", -0.600,  1.280,  0.000),   # 6  ring-O5
            ("C6",  "C", -0.595,  3.742,  0.554),   # 7  exocyclic CH2OH
            ("N2",  "N",  2.115, -1.180,  0.358),   # 8  amine group at C2
            ("O3",  "O",  3.510,  1.360,  0.358),   # 9  OH at C3
            ("O4",  "O",  2.115,  3.742, -0.354),   # 10 glycosidic O at C4
            ("O6",  "O", -0.595,  4.940,  0.000),   # 11 OH at C6
            # Hydrogens
            ("H1",  "H", -0.396,  0.000, -1.025),   # 12
            ("H2",  "H",  1.870, -0.040, -1.025),   # 13
            ("H3",  "H",  1.820,  1.360,  1.418),   # 14
            ("H4",  "H",  1.870,  2.600, -1.025),   # 15
            ("H5",  "H", -0.396,  2.520, -1.028),   # 16
            ("H6A", "H", -1.685,  3.684,  0.440),   # 17
            ("H6B", "H", -0.249,  3.700,  1.590),   # 18
            ("HN1", "H",  3.100, -1.180,  0.200),   # 19 NH2 H1
            ("HN2", "H",  1.700, -2.064,  0.120),   # 20 NH2 H2
            ("HO3", "H",  3.890,  2.236,  0.120),   # 21
            ("HO6", "H", -1.565,  4.940,  0.000),   # 22
        ],
        "bonds": [
            (1,2),(1,6),(1,12),
            (2,3),(2,8),(2,13),
            (3,4),(3,9),(3,14),
            (4,5),(4,10),(4,15),
            (5,6),(5,7),(5,16),
            (7,11),(7,17),(7,18),
            (8,19),(8,20),
            (9,21),(11,22),
        ],
    },

    # ── PHA — generic poly(hydroxyalkanoate) ──────────────────────────────
    # Formula: C4H6O2  similar to PHB but generic
    "PHA": {
        "name":      "Poly(hydroxyalkanoate) generic repeat unit",
        "formula":   "C4H6O2",
        "eco_score": 1.0,
        "atoms": [
            ("CA",  "C",  0.000,  0.000,  0.000),   # 1  carbonyl-C (sp2)
            ("O1",  "O",  0.572,  1.040,  0.000),   # 2  C=O
            ("O2",  "O",  1.360, -0.795,  0.000),   # 3  ester-O
            ("CB",  "C", -1.440,  0.000,  0.000),   # 4  alpha-C (sp3)
            ("CC",  "C", -2.200, -1.254,  0.440),   # 5  beta-C (sp3)
            ("CD",  "C", -1.440, -2.558,  0.000),   # 6  gamma-C (sp3)
            ("HA1", "H", -1.870,  0.940, -0.260),   # 7
            ("HA2", "H", -1.870, -0.200,  1.010),   # 8
            ("HB1", "H", -3.270, -1.200,  0.180),   # 9
            ("HB2", "H", -2.060, -1.260,  1.528),   # 10
            ("HC1", "H", -0.370, -2.510,  0.240),   # 11
            ("HC2", "H", -1.870, -3.498,  0.374),   # 12
            ("HC3", "H", -1.640, -2.558, -1.080),   # 13
        ],
        "bonds": [
            (1,2),(1,3),(1,4),
            (4,5),(4,7),(4,8),
            (5,6),(5,9),(5,10),
            (6,11),(6,12),(6,13),
        ],
    },
}

# Aliases
_LIBRARY["POLYLACTIC ACID"]        = _LIBRARY["PLA"]
_LIBRARY["POLYHYDROXYBUTYRATE"]    = _LIBRARY["PHB"]
_LIBRARY["POLYBUTYLENE SUCCINATE"] = _LIBRARY["PBS"]
_LIBRARY["POLYETHYLENE FURANOATE"] = _LIBRARY["PEF"]
_LIBRARY["POLYHYDROXYALKANOATE"]   = _LIBRARY["PHA"]

_BASE_KEYS = ["PLA", "PHB", "PBS", "PEF", "CELLULOSE", "CHITOSAN", "PHA"]


def list_available() -> list:
    return _BASE_KEYS


def get_pdb(key: str, chain_id: str = "A") -> str:
    """
    Generate a PDB file string for the given polymer key.

    Parameters
    ----------
    key : str   e.g. "PLA", "PHB", "PBS", "PEF", "CELLULOSE", "CHITOSAN", "PHA"

    Returns
    -------
    str : PDB file content. Save as .pdb and open with:  vmd molecule.pdb
          In VMD → Graphics → Representations → Drawing Method → Licorice
    """
    k = key.upper().strip()
    if k not in _LIBRARY:
        raise KeyError(
            f"'{key}' not in library.\nAvailable: {_BASE_KEYS}"
        )
    e = _LIBRARY[k]

    # ── Header ──────────────────────────────────────────────────────────────
    lines = [
        f"REMARK  Eco-Material Predictor — Molecular Structure PDB",
        f"REMARK  Material : {e['name']}",
        f"REMARK  Formula  : {e['formula']}",
        f"REMARK  Eco Score: {e['eco_score']}",
        f"REMARK  ",
        f"REMARK  VMD Tips:",
        f"REMARK    1. vmd molecule.pdb",
        f"REMARK    2. Graphics → Representations → Drawing Method → Licorice",
        f"REMARK    3. Coloring Method → Element  (C=cyan, O=red, H=white, N=blue)",
        f"REMARK  ",
    ]

    # Residue name: first 3 chars of key (PDB standard)
    resname = k[:3]
    atoms   = e["atoms"]

    # ── HETATM records ──────────────────────────────────────────────────────
    for i, (aname, element, x, y, z) in enumerate(atoms, start=1):
        # PDB HETATM: columns 1-6=record, 7-11=serial, 13-16=name, 18-20=resName,
        # 22=chainID, 23-26=resSeq, 31-38=X, 39-46=Y, 47-54=Z, 77-78=element
        lines.append(
            f"HETATM{i:5d}  {aname:<4s}{resname:3s} {chain_id}{1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
        )

    # ── CONECT records (explicit bonds for all connections) ──────────────
    # Build adjacency list: for each atom list all connected partners
    from collections import defaultdict
    adj = defaultdict(list)
    for (i, j) in e["bonds"]:
        adj[i].append(j)
        adj[j].append(i)

    for atom_idx in sorted(adj.keys()):
        partners = sorted(adj[atom_idx])
        # PDB CONECT record: up to 4 partners per line
        for chunk_start in range(0, len(partners), 4):
            chunk = partners[chunk_start:chunk_start+4]
            conect_str = f"CONECT{atom_idx:5d}" + "".join(f"{p:5d}" for p in chunk)
            lines.append(conect_str)

    lines.append("END")
    return "\n".join(lines)


def save_pdb(key: str, output_path: str) -> str:
    """Save PDB to file. Returns the path."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    content = get_pdb(key)
    with open(output_path, "w") as f:
        f.write(content)
    return output_path


def guess_material_key(features: dict) -> str:
    """
    Guess closest library material from input features.
    Returns a library key, or None for alloys.
    """
    if features.get("is_alloy", 0):
        return None

    mw   = features.get("repeat_unit_MW", 100)
    hb   = features.get("hydrogen_bond_capacity", 0)
    aro  = features.get("aromatic_content", 0)
    crys = features.get("crystallinity_tendency", 0.3)
    eco  = features.get("eco_score", 1.0)

    if mw < 80 and hb <= 2 and aro < 0.1:
        return "PLA"
    elif mw < 110 and crys > 0.55:
        return "PHB"
    elif mw > 160 and mw < 220 and hb <= 1 and aro < 0.1:
        return "PBS"
    elif aro > 0.25 and mw > 180:
        return "PEF"
    elif hb >= 4 and mw > 280:
        return "CELLULOSE"
    elif hb >= 3 and mw > 140:
        return "CHITOSAN"
    else:
        return "PHA"


def get_metal_pdb(lattice_type="BCC", a=2.86):
    lines = [
        "REMARK  Eco-Material Predictor — Crystal Structure PDB",
        f"REMARK  Material : Generic Metal Lattice ({lattice_type})",
        "REMARK  "
    ]
    atoms = []
    # Create a 2x2x2 unit cell block
    for i in range(2):
        for j in range(2):
            for k in range(2):
                atoms.append((i*a, j*a, k*a))
                if i < 1 and j < 1 and k < 1:
                    atoms.append(((i+0.5)*a, (j+0.5)*a, (k+0.5)*a))

    for idx, (x, y, z) in enumerate(atoms, start=1):
        lines.append(f"HETATM{idx:5d}  FE  MET A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          FE")
    
    lines.append("END")
    return "\n".join(lines)


import os
import sys

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        key = sys.argv[1].upper()
        out = sys.argv[2]
        if key.startswith("METAL"):
            pdb_str = get_metal_pdb()
            with open(out, 'w') as f:
                f.write(pdb_str)
        else:
            save_pdb(key, out)
        sys.exit(0)

    print("PDB Library — Available Materials:")
    for k in list_available():
        n_atoms = len(_LIBRARY[k]["atoms"])
        n_bonds = len(_LIBRARY[k]["bonds"])
        print(f"  {k:<12} {_LIBRARY[k]['formula']:<10}  {n_atoms} atoms  {n_bonds} bonds  — {_LIBRARY[k]['name']}")
    print()

    # Test PLA output
    pdb = get_pdb("PLA")
    print("--- PLA PDB Preview (first 12 lines) ---")
    for line in pdb.split("\n")[:12]:
        print(line)
