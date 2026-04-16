from __future__ import annotations

from Bio.PDB import MMCIFParser, PDBParser, MMCIFIO, PDBIO
from Bio.PDB.Structure import Structure
from pathlib import Path

from ..utils.logging_utils import Logger
import json
import tempfile
from ..utils.common_utils import convert_to_json_serializable, InlineJSONEncoder, wrap_leaf_lists_as_rawjson, get_clean_filename, get_optimized_filename

from ..utils.structure_utils import get_single_chain,get_residues_by_chain,get_sequence
from typing import List, Dict,Any, Tuple
import subprocess
from rdkit import Chem
from ..utils.substrate_utils import is_valid_mol_3d, build_docked_mol_from_atom_info
from Bio.PDB import StructureBuilder
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
import copy
import numpy as np



def file_exists(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() and p.is_file()

def get_stem(input_path: str | Path) -> str:
    return Path(input_path).stem

MAXFILENAME=150

def check_filename_length(name: str, logger: Logger) -> bool:
    if len(name) > MAXFILENAME:
        logger.print(f"[ERROR] Filename too long (>{MAXFILENAME}): {name}")
        return False
    return True

def load_protein_structure(path: str | Path, protein_name:str, logger: Logger) -> Structure | None:
    p = Path(path)

    try:
        if p.suffix.lower() in {".cif", ".mmcif"}:
            parser = MMCIFParser(QUIET=True)
        elif p.suffix.lower() == ".pdb":
            parser = PDBParser(QUIET=True)
        else:
            logger.print(f"[ERROR] Unsupported format: {p}")
            return None

        return parser.get_structure(protein_name, str(p))

    except Exception as e:
        logger.print(f"[ERROR] Exception in loading structure for {str(p)}: {e}")
        return None



def write_cif(struct: Structure, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    io = MMCIFIO()
    io.set_structure(struct)
    io.save(str(output_path))

def write_pdb(struct: Structure, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    io = PDBIO()
    io.set_structure(struct)
    io.save(str(output_path))

def write_json_from_dict(dict_data: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dict_data=convert_to_json_serializable(dict_data)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dict_data, f, indent=2, ensure_ascii=False)

def write_json_from_dict_inline_leaf_lists(dict_data: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dict_data = convert_to_json_serializable(dict_data)
    dict_data = wrap_leaf_lists_as_rawjson(dict_data)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            dict_data,
            f,
            cls=InlineJSONEncoder,
            indent=2,
            ensure_ascii=False
        )


def write_sdf(mol_3d: Chem.Mol, sdf_path: str | Path, logger: Logger,) -> bool:
    if not is_valid_mol_3d(mol_3d, logger):
        return False

    try:
        sdf_path = Path(sdf_path)
        sdf_path.parent.mkdir(parents=True, exist_ok=True)

        writer = Chem.SDWriter(str(sdf_path))
        conf_id = mol_3d.GetConformer().GetId()
        writer.write(mol_3d, confId=conf_id)
        writer.close()

        if not sdf_path.exists() or sdf_path.stat().st_size <= 0:
            logger.print("[ERROR] Failed to save SDF file.")
            return False

        return True
    except Exception:
        logger.print("[ERROR] Failed to save Mol(3D) to SDF file.")
        return False


def write_protein_pdbqt(struct: Structure,pdbqt_path: str | Path,logger: Logger) -> bool:
    try:
        pdbqt_path = Path(pdbqt_path)
        pdbqt_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_pdb = Path(tmp_dir) / "protein.pdb"

            # 写 PDB
            write_pdb(struct, tmp_pdb)

            if not tmp_pdb.exists() or tmp_pdb.stat().st_size <= 0:
                logger.print("[ERROR] Failed to write temporary PDB file.")
                return False

            # 调用 Meeko CLI
            p = subprocess.run(
                [
                    "mk_prepare_receptor.py",
                    "--read_pdb", str(tmp_pdb),
                    "--write_pdbqt", str(pdbqt_path),
                    "--allow_bad_res"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if p.returncode != 0:
                logger.print("[ERROR] mk_prepare_receptor.py failed.")
                return False

        if not pdbqt_path.exists() or pdbqt_path.stat().st_size <= 0:
            logger.print("[ERROR] Failed to generate PDBQT file.")
            return False

        return True

    except Exception:
        logger.print("[ERROR] Failed to convert Structure to PDBQT.")
        return False

def write_substrate_pdbqt_from_sdf(sdf_path: str | Path,pdbqt_path: str | Path,logger: Logger) -> bool:
    try:
        sdf_path = Path(sdf_path)
        pdbqt_path = Path(pdbqt_path)
        pdbqt_path.parent.mkdir(parents=True, exist_ok=True)

        if not sdf_path.exists() or sdf_path.stat().st_size <= 0:
            logger.print("[ERROR] Invalid input SDF file.")
            return False

        # 调用 Meeko CLI
        p = subprocess.run(
            [
                "mk_prepare_ligand.py",
                "-i", str(sdf_path),
                "-o", str(pdbqt_path),
                "--add_index_map",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if p.returncode != 0:
            logger.print("[ERROR] mk_prepare_ligand.py failed.")
            return False

        if not pdbqt_path.exists() or pdbqt_path.stat().st_size <= 0:
            logger.print("[ERROR] Failed to generate PDBQT file.")
            return False

        return True

    except Exception:
        logger.print("[ERROR] Failed to convert SDF to PDBQT.")
        return False

def load_sdf_mol_3d(sdf_path: str | Path, logger: Logger) -> Chem.Mol | None:
    try:
        sdf_path = Path(sdf_path)

        if not sdf_path.exists() or sdf_path.stat().st_size <= 0:
            logger.print("[ERROR] Invalid input SDF file.")
            return None

        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        if supplier is None or len(supplier) == 0:
            logger.print("[ERROR] Failed to load SDF file.")
            return None

        mol = supplier[0]
        if mol is None:
            logger.print("[ERROR] Failed to parse Mol from SDF file.")
            return None

        if mol.GetNumConformers() <= 0:
            logger.print("[ERROR] Input SDF does not contain 3D coordinates.")
            return None

        return mol

    except Exception:
        logger.print("[ERROR] Failed to read Mol(3D) from SDF file.")
        return None

def write_docked_sdf_from_atom_info(original_mol_3d: Chem.Mol,docked_atom_info_list: List[Dict[str, Any]],sdf_path: str | Path,logger: Logger) -> Chem.Mol | None:

    if original_mol_3d is None or original_mol_3d.GetNumConformers() <= 0:
        logger.print("[ERROR] Invalid original Mol(3D).")
        return None

    if not isinstance(docked_atom_info_list, list) or len(docked_atom_info_list) == 0:
        logger.print("[ERROR] Invalid docked_atom_info_list.")
        return None

    try:
        sdf_path = Path(sdf_path)
        sdf_path.parent.mkdir(parents=True, exist_ok=True)

        mol = build_docked_mol_from_atom_info(
            original_mol_3d,
            docked_atom_info_list,
            logger
        )
        if mol is None:
            return None

        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()

        if not sdf_path.exists() or sdf_path.stat().st_size <= 0:
            logger.print("[ERROR] Failed to save docked SDF file.")
            return None

        return mol

    except Exception:
        logger.print("[ERROR] Failed to write docked atom information to SDF file")
        return None

def write_docked_complex_from_mol_list(
    struct: Structure,
    docked_mol_list: List[Chem.Mol],
    protein_name: str,
    substrate_names: str,
    output_dir: str | Path,
    logger: Logger
) -> str | None:
    if struct is None:
        logger.print("[ERROR] struct is None.")
        return None

    if not isinstance(docked_mol_list, list) or len(docked_mol_list) == 0:
        logger.print("[ERROR] Invalid docked_mol_list.")
        return None

    if not isinstance(protein_name, str) or not protein_name.strip():
        logger.print("[ERROR] Invalid protein_name.")
        return None

    if not isinstance(substrate_names, str) or not substrate_names.strip():
        logger.print("[ERROR] Invalid substrate_names.")
        return None

    if not isinstance(output_dir, (str, Path)):
        logger.print("[ERROR] output_dir must be a str or Path.")
        return None

    try:
        protein_chain = get_single_chain(struct, logger)
        if protein_chain is None:
            return None

        residue_info_list = get_residues_by_chain(protein_chain, logger)
        if residue_info_list is None or len(residue_info_list) == 0:
            logger.print("[ERROR] Failed to get valid protein residues.")
            return None

        max_protein_resseq = max(res_id[1] for res_id, _, _ in residue_info_list)
        ligand_resseq_start = max_protein_resseq + 1

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        complex_name = f"docked_{protein_name}_{substrate_names}"
        complex_name = get_optimized_filename(complex_name)
        cif_path = output_dir / f"{complex_name}.cif"
        pdb_path = output_dir / f"{complex_name}.pdb"

        builder = StructureBuilder.StructureBuilder()
        builder.init_structure("complex")
        builder.init_model(0)
        builder.init_chain("A")

        new_struct = builder.get_structure()
        new_model: Model = new_struct[0]
        new_protein_chain: Chain = new_model["A"]

        for residue in protein_chain.get_residues():
            new_protein_chain.add(copy.deepcopy(residue))

        # === 计算 protein 当前最大 atom serial，ligand 从后面继续编号 ===
        max_serial = 0
        for atom in new_protein_chain.get_atoms():
            try:
                serial = int(atom.serial_number)
                if serial > max_serial:
                    max_serial = serial
            except Exception:
                continue

        current_serial = max_serial + 1

        ligand_chain = Chain("L")
        new_model.add(ligand_chain)

        for ligand_i, mol in enumerate(docked_mol_list, start=1):
            if mol is None or mol.GetNumConformers() <= 0:
                logger.print(f"[ERROR] Invalid docked Mol for ligand index {ligand_i}.")
                return None

            conf = mol.GetConformer()
            ligand_res_id = ("H", ligand_resseq_start + ligand_i - 1, " ")
            ligand_residue = Residue(ligand_res_id, "LIG", " ")

            element_count_dict: Dict[str, int] = {}

            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol().upper().strip()
                if not symbol:
                    symbol = "X"

                element_count_dict[symbol] = element_count_dict.get(symbol, 0) + 1
                atom_name = f"{symbol}{element_count_dict[symbol]}"
                atom_name = atom_name[:4]
                fullname = atom_name.rjust(4)

                pos = conf.GetAtomPosition(atom.GetIdx())
                serial_number = current_serial
                current_serial += 1

                pdb_atom = Atom(
                    name=atom_name,
                    coord=np.array([float(pos.x), float(pos.y), float(pos.z)], dtype=float),
                    bfactor=1.0,
                    occupancy=1.0,
                    altloc=" ",
                    fullname=fullname,
                    serial_number=serial_number,
                    element=symbol.capitalize(),
                )
                ligand_residue.add(pdb_atom)

            ligand_chain.add(ligand_residue)

        write_cif(new_struct, cif_path)
        write_pdb(new_struct, pdb_path)

        if not cif_path.exists() or cif_path.stat().st_size == 0:
            logger.print("[ERROR] Failed to write complex CIF.")
            return None

        if not pdb_path.exists() or pdb_path.stat().st_size == 0:
            logger.print("[ERROR] Failed to write complex PDB.")
            return None

        return str(cif_path)

    except Exception:
        logger.print(f"[ERROR] Failed to build docked complex CIF/PDB from Mol list")
        return None

