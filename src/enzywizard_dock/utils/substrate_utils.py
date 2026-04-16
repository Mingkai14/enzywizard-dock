from __future__ import annotations


from ..utils.logging_utils import Logger


from rdkit import Chem, DataStructs

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")



import math

from typing import Any, Dict, List, Optional, Set, Tuple

# Validation
def is_valid_smiles(smiles: str) -> bool:
    try:
        if not isinstance(smiles, str):
            return False
        smiles = smiles.strip()
        if not smiles:
            return False
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False

def is_valid_mol_2d(mol: Chem.Mol, logger: Logger) -> bool:
    if mol is None:
        logger.print("[ERROR] Input Mol(2D) is None.")
        return False

    try:
        if not isinstance(mol, Chem.Mol):
            logger.print("[ERROR] Input object is not an RDKit Mol(2D).")
            return False

        if mol.GetNumAtoms() <= 0:
            logger.print("[ERROR] Input Mol(2D) contains no atoms.")
            return False

        Chem.SanitizeMol(mol)
        return True
    except Exception:
        logger.print("[ERROR] Input Mol(2D) is invalid or failed sanitization.")
        return False


def is_valid_mol_h(mol_h: Chem.Mol, logger: Logger) -> bool:
    if mol_h is None:
        logger.print("[ERROR] Input Mol(H) is None.")
        return False

    try:
        if not isinstance(mol_h, Chem.Mol):
            logger.print("[ERROR] Input object is not an RDKit Mol(H).")
            return False

        if mol_h.GetNumAtoms() <= 0:
            logger.print("[ERROR] Input Mol(H) contains no atoms.")
            return False

        Chem.SanitizeMol(mol_h)

        has_h = any(atom.GetAtomicNum() == 1 for atom in mol_h.GetAtoms())
        if not has_h:
            logger.print("[ERROR] Input Mol(H) does not contain explicit hydrogen atoms.")
            return False

        return True
    except Exception:
        logger.print("[ERROR] Input Mol(H) is invalid or failed sanitization.")
        return False


def is_valid_conf_3d(conf: Chem.Conformer, logger: Logger) -> bool:
    if conf is None:
        logger.print("[ERROR] Input conformer is None.")
        return False

    try:
        if not isinstance(conf, Chem.Conformer):
            logger.print("[ERROR] Input object is not an RDKit Conformer.")
            return False

        if not conf.Is3D():
            logger.print("[ERROR] Input conformer is not 3D.")
            return False

        if conf.GetNumAtoms() <= 0:
            logger.print("[ERROR] Input conformer contains no atoms.")
            return False

        for atom_idx in range(conf.GetNumAtoms()):
            pos = conf.GetAtomPosition(atom_idx)
            if any(math.isnan(v) or math.isinf(v) for v in [pos.x, pos.y, pos.z]):
                logger.print("[ERROR] Input conformer contains invalid 3D coordinates.")
                return False

        return True
    except Exception:
        logger.print("[ERROR] Input conformer(3D) is invalid.")
        return False


def is_valid_mol_3d(mol_3d: Chem.Mol, logger: Logger) -> bool:
    if mol_3d is None:
        logger.print("[ERROR] Input Mol(3D) is None.")
        return False

    try:
        if not isinstance(mol_3d, Chem.Mol):
            logger.print("[ERROR] Input object is not an RDKit Mol(3D).")
            return False

        if mol_3d.GetNumAtoms() <= 0:
            logger.print("[ERROR] Input Mol(3D) contains no atoms.")
            return False

        Chem.SanitizeMol(mol_3d)

        if mol_3d.GetNumConformers() <= 0:
            logger.print("[ERROR] Input Mol(3D) contains no conformer.")
            return False

        conf = mol_3d.GetConformer()
        if not is_valid_conf_3d(conf, logger):
            return False

        if conf.GetNumAtoms() != mol_3d.GetNumAtoms():
            logger.print("[ERROR] Atom count mismatch between Mol(3D) and conformer.")
            return False

        return True
    except Exception:
        logger.print("[ERROR] Input Mol(3D) is invalid.")
        return False








































def build_docked_mol_from_atom_info(
    original_mol_3d: Chem.Mol,
    docked_atom_info_list: List[Dict[str, Any]],
    logger: Logger,
) -> Chem.Mol | None:

    if original_mol_3d is None or original_mol_3d.GetNumConformers() <= 0:
        logger.print("[ERROR] Invalid original Mol(3D).")
        return None

    if not isinstance(docked_atom_info_list, list) or len(docked_atom_info_list) == 0:
        logger.print("[ERROR] Invalid docked_atom_info_list.")
        return None

    try:
        atom_num = original_mol_3d.GetNumAtoms()

        used_original_atom_index_set = set()
        kept_original_atom_index_list: List[int] = []

        for item in docked_atom_info_list:
            original_atom_index = int(item.get("original_atom_index", 0))

            if original_atom_index <= 0 or original_atom_index > atom_num:
                logger.print("[ERROR] Invalid original atom index in docked_atom_info_list.")
                return None

            if original_atom_index in used_original_atom_index_set:
                logger.print("[ERROR] Duplicate original atom index.")
                return None

            used_original_atom_index_set.add(original_atom_index)
            kept_original_atom_index_list.append(original_atom_index)

        kept_original_atom_index_list.sort()

        old_to_new_index_dict = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(kept_original_atom_index_list)
        }

        rw_mol = Chem.RWMol()
        new_conf = Chem.Conformer(len(kept_original_atom_index_list))

        # ===== build atoms =====
        for old_index in kept_original_atom_index_list:
            old_atom = original_mol_3d.GetAtomWithIdx(old_index - 1)

            new_atom = Chem.Atom(old_atom.GetAtomicNum())
            new_atom.SetFormalCharge(old_atom.GetFormalCharge())
            new_atom.SetIsAromatic(old_atom.GetIsAromatic())
            new_atom.SetChiralTag(old_atom.GetChiralTag())
            new_atom.SetNoImplicit(old_atom.GetNoImplicit())
            new_atom.SetNumExplicitHs(old_atom.GetNumExplicitHs())
            new_atom.SetNumRadicalElectrons(old_atom.GetNumRadicalElectrons())

            rw_mol.AddAtom(new_atom)

        kept_old_index_set = set(kept_original_atom_index_list)

        # ===== build bonds =====
        for bond in original_mol_3d.GetBonds():
            b = bond.GetBeginAtomIdx() + 1
            e = bond.GetEndAtomIdx() + 1

            if b in kept_old_index_set and e in kept_old_index_set:
                rw_mol.AddBond(
                    old_to_new_index_dict[b],
                    old_to_new_index_dict[e],
                    bond.GetBondType(),
                )

        # ===== set coordinates =====
        for item in docked_atom_info_list:
            old_idx = int(item["original_atom_index"])
            x, y, z = float(item["x"]), float(item["y"]), float(item["z"])

            new_idx = old_to_new_index_dict[old_idx]
            new_conf.SetAtomPosition(new_idx, (x, y, z))

        mol = rw_mol.GetMol()
        mol.RemoveAllConformers()
        mol.AddConformer(new_conf, assignId=True)

        Chem.SanitizeMol(mol)

        return mol

    except Exception:
        logger.print("[ERROR] Failed to build docked Mol.")
        return None