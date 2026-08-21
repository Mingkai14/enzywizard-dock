[![DOI](https://zenodo.org/badge/1219037963.svg)](https://doi.org/10.5281/zenodo.19709478)
# Command: enzywizard-dock

EnzyWizard-Dock is a command-line tool for performing molecular docking
of one or multiple substrates with a cleaned protein structure and
generating a detailed JSON report.
It takes a cleaned CIF or PDB protein structure and substrate directory as input and performs docking
using AutoDock Vina. The tool supports both single-substrate docking and
simultaneous multi-substrate docking.
For each substrate, multiple conformations can be provided as
separate SDF files. The program automatically enumerates all possible
combinations of conformations of the same substrate and performs docking to identify
the optimal enzyme-substrate docking result.
By default, binding pockets are detected using PyVOL with a global docking box as fallback.
Alternatively, a docking box can be defined manually using a catalytic residue index or catalytic site coordinate together with a box size.
The tool outputs structured docking results suitable for downstream
applications such as enzyme-substrate interaction analysis, binding mode
evaluation, and enzyme characterising.


# Documentation index:

- example usage
- input parameters
- output files
- output report schema
- Process
- common errors and solutions
- dependencies
- references


# example usage:

The examples below use placeholder paths such as `path/to/input.cif`,
`path/to/substrate_dir/`, and `path/to/output_dir/`; replace them with your own
cleaned protein structure file, substrate SDF directory, and output directory.
Substrate names are provided with `-s` and must match SDF file names in
`--substrate_dir`. For example, `glucose` can match `glucose.sdf`,
`glucose_1.sdf`, `glucose_2.sdf`, and so on. The program searches the substrate
directory for all SDF files matching each substrate name and tries them one by
one during docking.

Dock a single substrate into a cleaned CIF structure using default settings.
By default, docking boxes are generated from PyVOL-detected pockets plus a
global protein box fallback, and Vina uses exhaustiveness 8.

```
enzywizard-dock -i path/to/input.cif -s "glucose" -d path/to/substrate_dir/ -o path/to/output_dir/
```

Dock a single substrate into a cleaned PDB structure.

```
enzywizard-dock -i path/to/input.pdb -s "glucose" -d path/to/substrate_dir/ -o path/to/output_pdb/
```

Dock two substrates simultaneously. Multiple substrate names are separated by
semicolons, and the program enumerates matched SDF conformer combinations for
each docking attempt.

```
enzywizard-dock -i path/to/input.cif -s "glucose;fructose" -d path/to/substrate_dir/ -o path/to/output_multi_substrate/
```

Use long option names for the same multi-substrate docking workflow.

```
enzywizard-dock --input_path path/to/input.cif --substrate_names "glucose;fructose" --substrate_dir path/to/substrate_dir/ --output_dir path/to/output_long_options/
```

Use a catalytic residue as the docking box center. The residue index is the
cleaned protein residue index, and the CA atom coordinate is used as the center.
This skips PyVOL pocket detection and the global docking box fallback, so it is
useful when a known catalytic residue or active-site residue is available. A
smaller box focuses the search and can run faster, but may miss valid poses
outside the box. A larger box explores a broader region, but can increase
runtime and reduce search precision at the same exhaustiveness.

```
enzywizard-dock -i path/to/input.cif -s "glucose" -d path/to/substrate_dir/ -o path/to/output_catalytic_residue/ --catalytic_residue 121 --box_size 20,20,20
```

Use an explicit catalytic site coordinate as the docking box center. This is
useful when the active-site coordinate is known from another analysis.

```
enzywizard-dock -i path/to/input.cif -s "glucose" -d path/to/substrate_dir/ -o path/to/output_site_coord/ --catalytic_site_coord 12.5,8.0,-3.2 --box_size 18,18,18
```

Use a higher Vina exhaustiveness for broader docking search. Larger values may
improve search coverage and docking robustness, but increase runtime. Smaller
values are faster but may miss better poses.

```
enzywizard-dock -i path/to/input.cif -s "glucose;fructose" -d path/to/substrate_dir/ -o path/to/output_high_exhaustiveness/ --exhaustiveness 32
```

Limit the number of docking attempts. A smaller value can reduce runtime, but
may stop before all candidate conformer and box combinations are tried.

```
enzywizard-dock -i path/to/input.cif -s "glucose;fructose" -d path/to/substrate_dir/ -o path/to/output_limited_attempts/ --max_docking_attempt_num 5
```

Disable early stopping so that the workflow continues after the first successful
docking result and keeps searching other conformer and box combinations up to
`--max_docking_attempt_num`. This can improve the chance of finding a lower
energy result, but increases runtime.

```
enzywizard-dock -i path/to/input.cif -s "glucose;fructose" -d path/to/substrate_dir/ -o path/to/output_no_early_stop/ --no_early_stop
```

Tune automatic pocket detection for smaller pockets. A smaller minimum volume
retains smaller predicted pockets and may increase the number of docking boxes.
This can improve coverage for small active sites, but may increase runtime and
include less relevant cavities.

```
enzywizard-dock -i path/to/input.cif -s "glucose" -d path/to/substrate_dir/ -o path/to/output_small_pockets/ --min_volume 25
```

Tune automatic pocket detection for broader pockets. Larger probe radii and a
larger minimum volume focus docking on broader, higher-volume pockets. This can
make the search more selective, but may miss narrow binding sites.

```
enzywizard-dock -i path/to/input.cif -s "glucose" -d path/to/substrate_dir/ -o path/to/output_broad_pockets/ --min_rad 2.0 --max_rad 8.0 --min_volume 200
```

Use multiple CPUs for Vina. A value of 0 lets Vina choose automatically; setting
a positive value can make runs more predictable on shared systems.

```
enzywizard-dock -i path/to/input.cif -s "glucose;fructose" -d path/to/substrate_dir/ -o path/to/output_cpu_4/ --cpu 4
```


# input parameters:

-i, --input_path
Required.
Path to the input cleaned protein structure file.
Supported file extensions: .cif, .pdb.

-s, --substrate_names
Required.
Input substrate names separated by ';'.
Each substrate name must match the SDF file name prefix in --substrate_dir.
The value is not parsed as a SMILES string by this command.

Examples:
- glucose
- glucose;fructose
- smiles1;smiles2

These examples match SDF files in --substrate_dir as follows:
- Input `glucose` matches `glucose.sdf`, `glucose_1.sdf`, `glucose_2.sdf`, ...
- Input `glucose;fructose` matches `glucose.sdf`, `glucose_*.sdf`, `fructose.sdf`, and `fructose_*.sdf` files with the supported naming pattern.
- Input `smiles1;smiles2` matches `smiles1.sdf`, `smiles1_*.sdf`, `smiles2.sdf`, and `smiles2_*.sdf` files with the supported naming pattern.

This parameter represents a multi-substrate combination for docking.

For input substrate_names "SubstrateA;SubstrateB", the program searches
substrate_dir for matched SDF files:

- SubstrateA.sdf
- SubstrateA_1.sdf
- SubstrateA_2.sdf
- ...

- SubstrateB.sdf
- SubstrateB_1.sdf
- SubstrateB_2.sdf
- ...

Each matched SDF file represents a different conformation.
The program automatically enumerates combinations such as:

- SubstrateA_1 + SubstrateB_1
- SubstrateA_1 + SubstrateB_2
- ...

and performs docking for each combination to identify the optimal result.

Duplicate substrate names are not allowed.

-d, --substrate_dir
Required.
Path to a directory containing input substrate SDF files.

-o, --output_dir
Required.
Directory to save docking outputs and the JSON report.

--max_docking_attempt_num
Optional.
Maximum number of docking attempts.

Default:
  20

Valid range:
  1 to 100

Smaller values can reduce runtime, but may stop before all candidate conformer
and docking box combinations are tried. Larger values allow more attempts when
many matched SDF files or docking boxes are available, but increase runtime.

--no_early_stop
Optional.
Disable stopping immediately after the first successful docking result.
By default, early stopping is enabled and the workflow returns after the first
successful docking result. Use this flag to keep trying additional conformer and
docking box combinations up to --max_docking_attempt_num. This may find a lower
energy result, but increases runtime.

--exhaustiveness
Optional.
Exhaustiveness of AutoDock Vina search.

Default:
  8

Valid range:
  1 to 64

Larger values may improve docking search coverage and robustness, but increase
runtime. Smaller values run faster but may miss better docking poses.

--cpu
Optional.
Number of CPUs used by AutoDock Vina.

Default:
  0

Valid range:
  0 or any positive integer

The default value 0 lets Vina decide automatically. A positive value sets the
number of CPUs used by Vina.

--min_rad
Optional.
Minimum probe radius used in pocket detection.

Default:
  1.8

Valid range:
  Greater than or equal to 1.2 and smaller than --max_rad

This parameter is only used when --catalytic_residue and --catalytic_site_coord
are not provided. Smaller values can detect narrower cavities, but excessively
small values may cause PyVOL/MSMS failure. Larger values ignore very narrow
cavities and focus on larger accessible pocket regions.

--max_rad
Optional.
Maximum probe radius used in pocket detection.

Default:
  6.2

Valid range:
  Greater than --min_rad

This parameter is only used when --catalytic_residue and --catalytic_site_coord
are not provided. Larger values allow broader pocket expansion, but excessively
large values may cause PyVOL/MSMS failure and can increase runtime. Smaller
values limit pocket expansion and may miss broader cavities.

--min_volume
Optional.
Minimum pocket volume threshold.

Default:
  50

Valid range:
  Greater than 20

This parameter is only used when --catalytic_residue and --catalytic_site_coord
are not provided. Smaller values retain smaller predicted pockets and may
increase the number of docking boxes. Larger values filter out smaller pockets
and focus docking on larger pocket regions.

--catalytic_residue
Optional.
Cleaned protein residue index used as the docking box center.

Example:
  121

This parameter is an integer residue index from the cleaned protein structure.
The CA atom coordinate of this residue is used as the docking box center.
When this parameter is provided, --box_size is required.
This parameter cannot be used together with --catalytic_site_coord.
When this parameter is provided, manual docking box mode is used. PyVOL pocket
detection and the global docking box fallback are not used.

--catalytic_site_coord
Optional.
Catalytic site center coordinate separated by ','.

Example:
  12.5,8.0,-3.2

When this parameter is provided, --box_size is required.
This parameter cannot be used together with --catalytic_residue.
When this parameter is provided, manual docking box mode is used. PyVOL pocket
detection and the global docking box fallback are not used.

--box_size
Optional.
Docking box size separated by ','.

Example:
  20,20,20

This parameter is required when --catalytic_residue or --catalytic_site_coord is provided.
All three values must be positive numbers.
Smaller boxes focus the search and can run faster, but may miss valid poses
outside the box. Larger boxes explore a broader region, but can increase runtime
and reduce search precision at the same exhaustiveness.


# output files:

The program outputs the following files into the output directory:

1. A JSON report
   - dock_report_{protein_name}_{substrate_names}.json
     - JSON report containing the selected enzyme-substrate docking result.
     - This file is generated only when a valid docking result is available.

2. Docked substrate structure files in SDF format
   - docked_{substrate_name}.sdf
     - Docked SDF file for each substrate in the selected docking result.
     - These files are generated only when a valid docking result is available.

3. Docked enzyme-substrate complex structure files
   - docked_{protein_name}_{substrate_names}.cif
     - Docked enzyme-substrate complex structure in CIF format.
   - docked_{protein_name}_{substrate_names}.pdb
     - Docked enzyme-substrate complex structure in PDB format.
   - These files are generated only when a valid docking result is available.

4. A log file
   - log.txt
     - Processing log containing informational messages and errors.


# output report schema:

The JSON report contains the following fields:

- "report_type"
  - Data type: string
  - Expected value: "enzywizard_dock"
  - Description: The field 'report_type' indicates the type of report ('report': http://purl.obolibrary.org/obo/IAO_0000088) generated by the EnzyWizard-Dock software.

- "enzyme_substrate_docking_result"
  - Data type: object
  - Description: The field 'enzyme_substrate_docking_result' indicates the docking result ('docking': https://goldbook.iupac.org/terms/view/11437) of substrates ('substrate': https://purl.dsmz.de/schema/Substrate) and the enzyme ('enzyme': https://purl.dsmz.de/schema/Enzyme) calculated by AutoDock Vina software ('AutoDock Vina': https://bio.tools/autodock_vina).

  The "enzyme_substrate_docking_result" object contains:

  - "enzyme_substrate_complex_name"
    - Data type: string
    - Description: The field 'enzyme_substrate_complex_name' indicates the name of the enzyme-substrate complex ('enzyme': https://purl.dsmz.de/schema/Enzyme; 'substrate': https://purl.dsmz.de/schema/Substrate; 'complex': https://goldbook.iupac.org/terms/view/C01203) generated by docking ('docking': https://goldbook.iupac.org/terms/view/11437).

  - "enzyme_substrate_binding_affinity"
    - Data type: number
    - Description: The field 'enzyme_substrate_binding_affinity' indicates the predicted binding affinity ('binding affinity': https://vina.scripps.edu/manual/#output) calculated by AutoDock Vina software ('AutoDock Vina': https://bio.tools/autodock_vina) from docking ('docking': https://goldbook.iupac.org/terms/view/11437) of the enzyme-substrate complex ('enzyme': https://purl.dsmz.de/schema/Enzyme; 'substrate': https://purl.dsmz.de/schema/Substrate; 'complex': https://goldbook.iupac.org/terms/view/C01203). Unit: kilocalories per mole (kcal/mol) ('kilocalorie': http://qudt.org/vocab/unit/KiloCAL; 'mole': http://qudt.org/vocab/unit/MOL).

  - "docked_substrate_names"
    - Data type: string
    - Description: The field 'docked_substrate_names' indicates the names of docked substrates ('substrate': https://purl.dsmz.de/schema/Substrate) in the enzyme-substrate complex ('enzyme': https://purl.dsmz.de/schema/Enzyme; 'substrate': https://purl.dsmz.de/schema/Substrate; 'complex': https://goldbook.iupac.org/terms/view/C01203).

  - "docking_box_center_coordinate"
    - Data type: array
    - Item data type: number
    - Number of items: 3
    - Description: The field 'docking_box_center_coordinate' indicates the center coordinate ('coordinate': https://mathworld.wolfram.com/Coordinates.html) of the docking box ('box': https://www.pbr-book.org/3ed-2018/Geometry_and_Transformations/Bounding_Boxes) used for docking ('docking': https://goldbook.iupac.org/terms/view/11437). Unit: angstroms (Å) ('angstrom': http://qudt.org/vocab/unit/ANGSTROM).

  - "docking_box_size"
    - Data type: array
    - Item data type: number
    - Number of items: 3
    - Description: The field 'docking_box_size' indicates the size ('size': http://purl.obolibrary.org/obo/PATO_0000117) of the docking box ('box': https://www.pbr-book.org/3ed-2018/Geometry_and_Transformations/Bounding_Boxes) used for docking ('docking': https://goldbook.iupac.org/terms/view/11437). Unit: angstroms (Å) ('angstrom': http://qudt.org/vocab/unit/ANGSTROM).

  - "docked_substrates"
    - Data type: array
    - Minimum number of items: 1
    - Description: The field 'docked_substrates' indicates the docked substrates ('substrate': https://purl.dsmz.de/schema/Substrate) in the enzyme-substrate complex ('enzyme': https://purl.dsmz.de/schema/Enzyme; 'substrate': https://purl.dsmz.de/schema/Substrate; 'complex': https://goldbook.iupac.org/terms/view/C01203) generated by docking ('docking': https://goldbook.iupac.org/terms/view/11437).

    Each item in "docked_substrates" is an object containing:

    - "docked_substrate_name"
      - Data type: string
      - Description: The field 'docked_substrate_name' indicates the name of the docked substrate ('substrate': https://purl.dsmz.de/schema/Substrate).

    - "docked_substrate_structure_name"
      - Data type: string
      - Description: The field 'docked_substrate_structure_name' indicates the name of the docked molecular structure ('molecular structure': http://edamontology.org/data_0883) of the substrate ('substrate': https://purl.dsmz.de/schema/Substrate).

    - "docked_substrate_center_coordinate"
      - Data type: array
      - Item data type: number
      - Number of items: 3
      - Description: The field 'docked_substrate_center_coordinate' indicates the center coordinate ('coordinate': https://mathworld.wolfram.com/Coordinates.html) of the docked substrate ('substrate': https://purl.dsmz.de/schema/Substrate) in the enzyme-substrate complex ('enzyme': https://purl.dsmz.de/schema/Enzyme; 'substrate': https://purl.dsmz.de/schema/Substrate; 'complex': https://goldbook.iupac.org/terms/view/C01203). Unit: angstroms (Å) ('angstrom': http://qudt.org/vocab/unit/ANGSTROM).

# Process:

This command processes the input cleaned protein structure as follows:

1. Load the input structure
   - Read the cleaned CIF or PDB file using Biopython (Bio.PDB).
   - Resolve the protein name from the input filename.

2. Validate input conditions
   - Check that the input file exists.
   - Validate that the structure satisfies the cleaned-structure requirement.

3. Determine docking box mode
   - If --catalytic_residue is provided, use the CA coordinate of that cleaned protein residue as the docking box center.
   - If --catalytic_site_coord is provided, use that coordinate as the docking box center.
   - In either manual docking box mode, use --box_size as the docking box size and skip PyVOL pocket detection and the global docking box fallback.
   - If no manual docking box parameter is provided, continue with automatic pocket detection.

4. Detect pocket regions
   - In automatic docking box mode, use PyVOL to detect pocket regions from the protein structure.

5. Compute global docking box
   - In automatic docking box mode, calculate a bounding box covering the entire protein structure.

6. Parse substrate inputs
   - Split substrate_names by ';' to obtain substrate list.

7. Search substrate files
   - Locate matched SDF files for each substrate in substrate_dir.

8. Enumerate substrate conformations
   - Treat multiple SDF files of the same substrate as alternative conformations.
   - Generate all combinations of substrate conformations.

9. Prepare docking inputs
   - Convert protein structure to receptor PDBQT format.
   - Convert each substrate SDF to ligand PDBQT format.

10. Build docking boxes
   - In automatic docking box mode, use pocket-based boxes and add one global structure box.
   - In manual docking box mode, use one user-defined box.

11. Perform docking
   - Iterate over substrate combinations and docking boxes.
   - Perform AutoDock Vina docking for each case.

12. Parse docking results
   - Extract docking poses and predicted binding affinities.
   - Map docked coordinates back to original ligand atoms.

13. Select best result
   - Choose the docking result with lowest predicted binding affinity.
   - Optionally stop early if early_stop=True.
   
14. Save docking outputs
   - Write docked substrate SDF files.
   - Generate enzyme-substrate complex CIF and PDB files.

15. Generate report
   - Save structured JSON report summarizing the enzyme-substrate docking result.


# common errors and solutions:

- "Invalid docking parameters. Require: max_docking_attempt_num (1–100), exhaustiveness (1–64)."
  - Cause: `--max_docking_attempt_num` or `--exhaustiveness` is outside the supported range.
  - Solution: Use `--max_docking_attempt_num` from 1 to 100 and `--exhaustiveness` from 1 to 64.

- "Input not found"
  - Cause: The path passed to `-i` or `--input_path` does not exist or is not a file.
  - Solution: Check the input path and make sure it points to an existing cleaned CIF or PDB file.

- "Invalid substrate_dir"
  - Cause: The path passed to `-d` or `--substrate_dir` does not exist or is not a directory.
  - Solution: Check that `--substrate_dir` points to a directory containing input SDF files.

- "Unsupported format"
  - Cause: The input structure extension is not `.cif` or `.pdb`.
  - Solution: Use a supported cleaned structure file format.

- "Exception in loading structure for"
  - Cause: Biopython could not parse the input file as a usable structure.
  - Solution: Check that the file is valid, non-empty, non-corrupted, and matches its file extension.

- "Structure must contain exactly one model. Please run 'enzywizard clean' first."
  - Cause: The input structure contains zero models or multiple models.
  - Solution: Run the structure through `enzywizard-clean` first and use the cleaned output as input.

- "Structure must contain exactly one chain. Please run 'enzywizard clean' first."
  - Cause: The input structure contains zero chains or multiple chains.
  - Solution: Run the structure through `enzywizard-clean` first and use the cleaned output as input.

- "Cleaned structure must use chain ID 'A'. Please run 'enzywizard clean' first."
  - Cause: The input is not in the cleaned single-chain format expected by EnzyWizard-Dock.
  - Solution: Use the cleaned output generated by `enzywizard-clean`.

- "No SDF files were found for substrate"
  - Cause: No SDF file in `--substrate_dir` matches the requested substrate name.
  - Solution: Make sure the substrate name matches SDF file names such as `glucose.sdf`, `glucose_1.sdf`, or `glucose_2.sdf`.

- "Duplicate substrate names are not allowed."
  - Cause: The same substrate name appears more than once in `--substrate_names`.
  - Solution: Remove duplicate names from the semicolon-separated substrate list.

- "Invalid substrate SDF file"
  - Cause: A matched SDF file is missing, empty, or not usable for docking.
  - Solution: Check that all matched SDF files exist, are non-empty, and contain valid 3D substrate structures.

- "Input SDF does not contain 3D coordinates."
  - Cause: A substrate SDF file has no 3D conformer coordinates.
  - Solution: Generate substrate SDF files with 3D conformers, for example using `enzywizard-substrate`, then rerun docking.

- "Failed to generate PDBQT file."
  - Cause: Meeko failed to generate receptor or ligand PDBQT input for Vina.
  - Solution: Check that Meeko is installed and available, then inspect the cleaned protein and substrate SDF files for unsupported or invalid chemistry.

- "mk_prepare_receptor.py failed with return code"
  - Cause: Meeko receptor preparation started but failed while converting the cleaned protein structure to PDBQT.
  - Solution: Review the output tail in `log.txt`, confirm Meeko is installed correctly, and check that the protein input is a valid cleaned structure.

- "mk_prepare_ligand.py failed with return code"
  - Cause: Meeko ligand preparation started but failed while converting a substrate SDF file to PDBQT.
  - Solution: Review the output tail in `log.txt` and check that the substrate SDF file is a valid 3D small-molecule structure.

- "Vina docking failed for"
  - Cause: AutoDock Vina failed during docking for a substrate combination and docking box.
  - Solution: Check that Vina is installed and available, review the Vina error in `log.txt`, and consider using valid 3D SDF inputs, a larger manual box, or a different pocket parameter set.

- "Vina docking output is empty for"
  - Cause: Vina completed without returning a usable pose for that substrate combination and box.
  - Solution: Check the substrate SDF quality and docking box placement. For manual boxes, increase or reposition the box if the active site is likely outside the current region.

- "No valid docking results were found for any substrate combination and docking box."
  - Cause: All attempted substrate conformer and docking-box combinations failed or produced no usable docking pose.
  - Solution: Check Vina, Meeko, substrate SDF quality, and docking box settings. If needed, increase `--max_docking_attempt_num` or use `--no_early_stop` to try more combinations.

- "Failed to write docked atom information to SDF file"
  - Cause: The selected docked ligand coordinates could not be written to SDF, usually because of an invalid molecule or filesystem problem.
  - Solution: Check output directory permissions, available disk space, and earlier ligand parsing messages in `log.txt`.

- "Failed to build docked complex CIF/PDB from Mol list."
  - Cause: The docked enzyme-substrate complex could not be assembled or written after docking.
  - Solution: Check whether the docked molecules were generated successfully and whether the output directory is writable.

- "Failed to write report JSON to"
  - Cause: The JSON report could not be written to the output directory because of a filesystem, permission, path, or disk-space problem.
  - Solution: Check that the `-o` output directory path is writable and that there is enough disk space.

- "--catalytic_residue and --catalytic_site_coord cannot be used together."
  - Cause: Two manual docking-box center definitions were provided at the same time.
  - Solution: Use either `--catalytic_residue` or `--catalytic_site_coord`, not both.

- "--box_size is required when --catalytic_residue or --catalytic_site_coord is provided."
  - Cause: Manual docking-box mode needs both a center and a box size.
  - Solution: Add `--box_size` with three positive comma-separated values, such as `20,20,20`.

- "--box_size values must be positive."
  - Cause: At least one docking box size value is zero or negative.
  - Solution: Use three positive numeric values for `--box_size`.

- "Invalid pocket detection parameters. Require: min_rad ≥ 1.2, max_rad > min_rad, min_volume > 20."
  - Cause: Automatic pocket-detection parameters are outside the supported range.
  - Solution: Use standard values such as `--min_rad 1.8 --max_rad 6.2 --min_volume 50`, then adjust one parameter at a time.

- "Failed to run PyVOL"
  - Cause: The PyVOL executable could not be started, usually because PyVOL is not installed or is not available on `PATH`.
  - Solution: Install PyVOL and confirm that the `pyvol` command can be run from the same environment.

- "PyVOL failed with return code"
  - Cause: PyVOL started but returned an error. Common causes include unsuitable probe-radius settings, problematic cleaned input geometry, or a PyVOL/MSMS environment issue.
  - Solution: Review the PyVOL output tail in `log.txt`, try default parameters, and then adjust `--min_rad`, `--max_rad`, or `--min_volume` gradually.

- Cleaned structure validation failed
  - Cause: The input is not a valid EnzyWizard-cleaned single-chain protein structure. Common causes include multiple chains, non-chain-A input, heterogens, insertion codes, non-standard residues, missing atoms, unexpected atoms, invalid occupancies, or non-continuous numbering.
  - Solution: Review the specific validation error above this summary in `log.txt`, run `enzywizard-clean`, and use its cleaned CIF or PDB output.

- No docked SDF or complex files are generated
  - Cause: No valid docking result was produced, so the command fails before writing the JSON report and docked structure files.
  - Solution: Review `log.txt` for Vina, Meeko, substrate SDF, or docking-box errors, then rerun after fixing the first reported cause.

- Output files are missing
  - Cause: The command failed before all outputs were written, no valid docking result was produced, or the output directory is not the one expected.
  - Solution: Check `log.txt`, confirm the `-o` output directory, and rerun after fixing earlier errors.

- Output file names are different from expected
  - Cause: Output names are derived from the input structure name and substrate names, then cleaned for filesystem compatibility. Characters such as semicolons, commas, spaces, colons, equals signs, and plus signs are converted to underscores.
  - Solution: Check the input file name and substrate names, then look for files named `dock_report_...json`, `docked_...sdf`, `docked_...cif`, and `docked_...pdb`.


# dependencies:

- AutoDock Vina
- Meeko
- RDKit
- Biopython
- PyVOL
- MSMS


# references:

- Eberhardt et al., AutoDock Vina 1.2.0
  https://doi.org/10.1021/acs.jcim.1c00203

- Trott & Olson, AutoDock Vina
  https://doi.org/10.1002/jcc.21334

- AutoDock Vina:
  https://vina.scripps.edu/

- Meeko:
  https://github.com/forlilab/Meeko

- RDKit:
  https://www.rdkit.org/

- Biopython:
  https://biopython.org/

- PyVOL:
  https://github.com/schlessinger-lab/pyvol

- MSMS:
  https://doi.org/10.1002/jcc.540150805
