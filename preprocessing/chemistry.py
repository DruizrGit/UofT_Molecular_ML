"""
Chemistry Features
"""
import numpy as np
import rdkit
from rdkit.Chem import Descriptors as rdcd
from rdkit.Chem import rdMolDescriptors as rdmol

def create_rdkit_molecule(
        df, 
        smiles_col = 'smiles', 
        output_col = 'molecule'
        ):
    """
    Create column with rdkit molecule objects
    """
    MolFromSmilesVec = np.vectorize(rdkit.Chem.MolFromSmiles)

    df[output_col] = MolFromSmilesVec(df[smiles_col])
    
    return df

def compute_radical_electrons(
        df, 
        molecule_col = 'molecule', 
        output_col = 'radical_electrons'
        ):
    """
    Compute number of radical electrons in molecule (unpaired electrons)
    """
    NumRadicalElectronsVec = np.vectorize(rdcd.NumRadicalElectrons)

    df[output_col] = NumRadicalElectronsVec(df[molecule_col])

    return df

def compute_exact_molecular_weight(
        df, 
        molecule_col = 'molecule',
        output_col = 'total_molecular_weight'
        ):
    """
    Compute molecular weight of molecule
    """
    ExactMolWtVec = np.vectorize(rdcd.ExactMolWt)

    df[output_col] = ExactMolWtVec(df[molecule_col])

    return df

def compute_avg_molecular_weight_no_hydrogen(
        df, 
        molecule_col = 'molecule',
        output_col = 'avg_molecule_weight_no_hydrogens'
        ):
    """
    Compute average molecular weight of molecule ignoring hydrogens
    """
    HeavyAtomMolWtVec = np.vectorize(rdcd.HeavyAtomMolWt)

    df[output_col] = HeavyAtomMolWtVec(df[molecule_col])

    return df

def compute_valence_electrons(
        df, 
        molecule_col = 'molecule',
        output_col = 'valence_electrons'
        ):
    """
    Compute number of valence electrons for the molecule
    """
    NumValenceElectronsVec = np.vectorize(rdcd.NumValenceElectrons)

    df[output_col] = NumValenceElectronsVec(df[molecule_col])

    return df

def compute_avg_molecular_weight(
        df, 
        molecule_col = 'molecule',
        output_col = 'avg_molecule_weight'
        ):
    """
    Compute average molecular weight of molecule
    """
    MolWtVec = np.vectorize(rdcd.MolWt)

    df[output_col] = MolWtVec(df[molecule_col])

    return df

def compute_max_partial_charge(
        df, 
        molecule_col = 'molecule',
        output_col = 'max_partial_charge'
        ):
    """
    Compute maximum partial charge
    """
    MaxPartialChargeVec = np.vectorize(rdcd.MaxPartialCharge)

    df[output_col] = MaxPartialChargeVec(df[molecule_col])

    return df

def compute_min_partial_charge(
        df, 
        molecule_col = 'molecule',
        output_col = 'min_partial_charge'
        ):
    """
    Compute minimum partial charge
    """
    MinPartialChargeVec = np.vectorize(rdcd.MinPartialCharge)

    df[output_col] = MinPartialChargeVec(df[molecule_col])

    return df

def compute_aliphatic_carbocycles(
        df, 
        molecule_col = 'molecule',
        output_col = 'aliphatic_carbocycles'
        ):
    """
    Compute number of aliphatic (containing at least one non-aromatic bond) carbocycles for a molecule.
    * Carbocycle is a ring containing only carbon atoms.
    """
    CalcNumAliphaticCarbocyclesVec = np.vectorize(rdmol.CalcNumAliphaticCarbocycles)

    df[output_col] = CalcNumAliphaticCarbocyclesVec(df[molecule_col])

    return df

def compute_aliphatic_heterocycles(
        df, 
        molecule_col = 'molecule',
        output_col = 'aliphatic_heterocycles'
        ):
    """
    Compute number of aliphatic (containing at least one non-aromatic bond) heterocycles for a molecule.
    * Heterocycle is a ring containing at least two different elements as members of its ring.
    """
    CalcNumAliphaticHeterocyclesVec = np.vectorize(rdmol.CalcNumAliphaticHeterocycles)
    
    df[output_col] = CalcNumAliphaticHeterocyclesVec(df[molecule_col])

    return df

def compute_aromatic_carbocycles(
        df, 
        molecule_col = 'molecule',
        output_col = 'aromatic_carbocycles'
        ):
    """
    Compute number of aliphatic carbocycles for a molecule.
    * Carbocycle is a ring containing only carbon atoms.
    """
    CalcNumAromaticCarbocyclesVec = np.vectorize(rdmol.CalcNumAromaticCarbocycles)

    df[output_col] = CalcNumAromaticCarbocyclesVec(df[molecule_col])

    return df

def compute_aromatic_heterocycles(
        df, 
        molecule_col = 'molecule',
        output_col = 'aromatic_heterocycles'
        ):
    """
    Compute number of aromatic heterocycles for a molecule.
    * Heterocycle is a ring containing at least two different elements as members of its ring.
    """
    CalcNumAromaticHeterocyclesVec = np.vectorize(rdmol.CalcNumAromaticHeterocycles)

    df[output_col] = CalcNumAromaticHeterocyclesVec(df[molecule_col])

    return df

def compute_amide_bonds(
        df, 
        molecule_col = 'molecule',
        output_col = 'amide_bonds'
        ):
    """
    Compute number of amide bonds in the molecule.
    """
    CalcNumAmideBondsVec = np.vectorize(rdmol.CalcNumAmideBonds)

    df[output_col] = CalcNumAmideBondsVec(df[molecule_col])

    return df

def compute_num_atoms(
        df, 
        molecule_col = 'molecule',
        output_col = 'num_atoms'
        ):
    """
    Compute total number of atoms in the molecule.
    """
    CalcNumAtomsVec = np.vectorize(rdmol.CalcNumAtoms)

    df[output_col] = CalcNumAtomsVec(df[molecule_col])

    return df

def compute_bridgehead_atoms(
        df, 
        molecule_col = 'molecule',
        output_col = 'bridgehead_atoms'
        ):
    """
    Compute total number of bridgehead atoms (atoms shared between rings that share at least two bonds) in the molecule
    """
    CalcNumBridgeheadAtomsVec = np.vectorize(rdmol.CalcNumBridgeheadAtoms)

    df[output_col] = CalcNumBridgeheadAtomsVec(df[molecule_col])

    return df

def compute_hbond_acceptors(
        df, 
        molecule_col = 'molecule',
        output_col = 'hbond_acceptors'
        ):
    """
    Compute number of H-bond acceptors for the molecule.
    """
    CalcNumHBAVec = np.vectorize(rdmol.CalcNumHBA)

    df[output_col] = CalcNumHBAVec(df[molecule_col])

    return df

def compute_hbond_donors(
        df, 
        molecule_col = 'molecule',
        output_col = 'hbond_donors'
        ):
    """
    Compute number of H-bond donors for the molecule.
    """
    CalcNumHBDVec = np.vectorize(rdmol.CalcNumHBD)
    
    df[output_col] = CalcNumHBDVec(df[molecule_col])

    return df

def compute_lipinski_hbond_acceptors(
        df, 
        molecule_col = 'molecule',
        output_col = 'lipinski_hbond_acceptors'
        ):
    """
    Compute number of Lipinski H-bond acceptors for the molecule.
    """
    CalcNumLipinskiHBAVec = np.vectorize(rdmol.CalcNumLipinskiHBA)
    
    df[output_col] = CalcNumLipinskiHBAVec(df[molecule_col])

    return df

def compute_lipinski_hbond_donors(
        df, 
        molecule_col = 'molecule',
        output_col = 'lipinski_hbond_donors'
        ):
    """
    Compute number of Lipinski H-bond donors for the molecule.
    """
    CalcNumLipinskiHBDVec = np.vectorize(rdmol.CalcNumLipinskiHBD)

    df[output_col] = CalcNumLipinskiHBDVec(df[molecule_col])

    return df

def compute_heavy_atoms(
        df, 
        molecule_col = 'molecule',
        output_col = 'heavy_atoms'
        ):
    """
    Compute number of heavy atoms for the molecule.
    * Heavy Atom is any atom that is not hydrogen.
    """
    CalcNumHeavyAtomsVec = np.vectorize(rdmol.CalcNumHeavyAtoms)

    df[output_col] = CalcNumHeavyAtomsVec(df[molecule_col])

    return df

def compute_hetero_atoms(
        df, 
        molecule_col = 'molecule',
        output_col = 'hetero_atoms'
        ):
    """
    Compute number of heteroatoms for the molecule.
    * Heteroatoms are any atom that is not hydrogen or carbon.
    """
    CalcNumHeteroatomsVec = np.vectorize(rdmol.CalcNumHeteroatoms)

    df[output_col] = CalcNumHeteroatomsVec(df[molecule_col])

    return df

def compute_rotatable_bonds(
        df, 
        molecule_col = 'molecule',
        output_col = 'rotatable_bonds'
        ):
    """
    Compute number of rotatable bonds for the molecule.
    """
    CalcNumRotatableBondsVec = np.vectorize(rdmol.CalcNumRotatableBonds)

    df[output_col] = CalcNumRotatableBondsVec(df[molecule_col])

    return df

def compute_spiro_atoms(
        df, 
        molecule_col = 'molecule',
        output_col = 'spiro_atoms'
        ):
    """
    Compute number of spiro atoms (atoms shared between rings that share exactly one atom)
    """
    CalcNumSpiroAtomsVec = np.vectorize(rdmol.CalcNumSpiroAtoms)

    df[output_col] = CalcNumSpiroAtomsVec(df[molecule_col])

    return df











