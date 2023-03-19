"""
Chemistry Features
"""
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
    df[output_col] = df[smiles_col].apply(lambda mol: rdkit.Chem.MolFromSmiles(mol))

    return df

def compute_radical_electrons(
        df, 
        molecule_col = 'molecule', 
        output_col = 'radical_electrons'
        ):
    """
    Compute number of radical electrons in molecule (unpaired electrons)
    """
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdcd.NumRadicalElectrons(mol))

    return df

def compute_exact_molecular_weight(
        df, 
        molecule_col = 'molecule',
        output_col = 'total_molecular_weight'
        ):
    """
    Compute molecular weight of molecule
    """
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdcd.ExactMolWt(mol))

    return df

def compute_avg_molecular_weight_no_hydrogen(
        df, 
        molecule_col = 'molecule',
        output_col = 'avg_molecule_weight_no_hydrogens'
        ):
    """
    Compute average molecular weight of molecule ignoring hydrogens
    """
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdcd.HeavyAtomMolWt(mol))

    return df

def compute_valence_electrons(
        df, 
        molecule_col = 'molecule',
        output_col = 'valence_electrons'
        ):
    """
    Compute number of valence electrons for the molecule
    """
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdcd.NumValenceElectrons(mol))

    return df

def compute_avg_molecular_weight(
        df, 
        molecule_col = 'molecule',
        output_col = 'avg_molecule_weight'
        ):
    """
    Compute average molecular weight of molecule
    """
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdcd.MolWt(mol))

    return df

def compute_max_partial_charge(
        df, 
        molecule_col = 'molecule',
        output_col = 'max_partial_charge'
        ):
    """
    Compute maximum partial charge
    """
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdcd.MaxPartialCharge(mol))

    return df

def compute_min_partial_charge(
        df, 
        molecule_col = 'molecule',
        output_col = 'min_partial_charge'
        ):
    """
    Compute minimum partial charge
    """
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdcd.MinPartialCharge(mol))

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
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdmol.CalcNumAliphaticCarbocycles(mol))

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
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdmol.CalcNumAliphaticHeterocycles(mol))

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
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdmol.CalcNumAromaticCarbocycles(mol))

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
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdmol.CalcNumAromaticHeterocycles(mol))

    return df

def compute_amide_bonds(
        df, 
        molecule_col = 'molecule',
        output_col = 'amide_bonds'
        ):
    """
    Compute number of amide bonds in the molecule.
    """
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdmol.CalcNumAmideBonds(mol))

    return df

def compute_num_atoms(
        df, 
        molecule_col = 'molecule',
        output_col = 'num_atoms'
        ):
    """
    Compute total number of atoms in the molecule.
    """
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdmol.CalcNumAtoms(mol))

    return df

def compute_bridgehead_atoms(
        df, 
        molecule_col = 'molecule',
        output_col = 'bridgehead_atoms'
        ):
    """
    Compute total number of bridgehead atoms (atoms shared between rings that share at least two bonds) in the molecule
    """
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdmol.CalcNumBridgeheadAtoms(mol))

    return df

def compute_hbond_acceptors(
        df, 
        molecule_col = 'molecule',
        output_col = 'hbond_acceptors'
        ):
    """
    Compute number of H-bond acceptors for the molecule.
    """
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdmol.CalcNumHBA(mol))

    return df

def compute_hbond_donors(
        df, 
        molecule_col = 'molecule',
        output_col = 'hbond_donors'
        ):
    """
    Compute number of H-bond donors for the molecule.
    """
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdmol.CalcNumHBD(mol))

    return df

def compute_lipinski_hbond_acceptors(
        df, 
        molecule_col = 'molecule',
        output_col = 'lipinski_hbond_acceptors'
        ):
    """
    Compute number of Lipinski H-bond acceptors for the molecule.
    """
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdmol.CalcNumLipinskiHBA(mol))

    return df

def compute_lipinski_hbond_donors(
        df, 
        molecule_col = 'molecule',
        output_col = 'lipinski_hbond_donors'
        ):
    """
    Compute number of Lipinski H-bond donors for the molecule.
    """
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdmol.CalcNumLipinskiHBD(mol))

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
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdmol.CalcNumHeavyAtoms(mol))

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
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdmol.CalcNumHeteroatoms(mol))

    return df

def compute_rotatable_bonds(
        df, 
        molecule_col = 'molecule',
        output_col = 'rotatable_bonds'
        ):
    """
    Compute number of rotatable bonds for the molecule.
    """
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdmol.CalcNumRotatableBonds(mol))

    return df

def compute_spiro_atoms(
        df, 
        molecule_col = 'molecule',
        output_col = 'spiro_atoms'
        ):
    """
    Compute number of spiro atoms (atoms shared between rings that share exactly one atom)
    """
    
    df[output_col] = df[molecule_col].apply(lambda mol: rdmol.CalcNumSpiroAtoms(mol))

    return df











