
from collections import defaultdict

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from .smiles_utils import postprocess, canonicalize_smiles

fingerprint_prop_key = 'atomNote'


def group_invariant(mol):
    """consider halogen atoms as same

    Args:
        mol (Mol): rdkit.Chem.rdchem.Mol

    Returns:
        list: list of int which indicate the types of atoms.
    """

    import ctypes
    invariant_list = []
    for a in mol.GetAtoms():
        atomic_num = a.GetAtomicNum()
        if atomic_num == 17 or atomic_num == 35 or atomic_num == 53:
            f = (9, a.GetDegree(), a.GetFormalCharge())
        else:
            f = (atomic_num, a.GetDegree(), a.GetFormalCharge(), a.IsInRing())
        f = ctypes.c_uint32(hash(f)).value
        invariant_list.append(f)
    return invariant_list


def calculate_mol_fp(mol, radius, addfp2prop=True):
    """Calculates the Morgan fingerprint and bitinfo of a molecule

    Args:
        mol (Mol): rdkit.Chem.rdchem.Mol
        radius (int): the maximum radius of Morgan fingerprint        
        addfp2prop (bool, optional): store fp value to atom prop. Defaults to True.

    Returns:
        Tuple: (a Morgan fingerprint for a molecule, the corresponding bit information)
    """

    fp_info = {}
    invariant_list = group_invariant(mol)
    fp = AllChem.GetMorganFingerprint(mol,
                                      radius=radius,
                                      bitInfo=fp_info,
                                      invariants=invariant_list,
                                      useChirality=True)
    if addfp2prop:
        # add fingerprint value to property, only for radius within 2
        add_fp_prop(mol, fp_info=fp_info)
    return fp_info


def calculate_mols_fps(mol_list, radius):
    """Calculates the Morgan fingerprint and bitinfo for a list of molecules

    Args:
        mol_list (list): a list of molecules (rdkit.Chem.rdchem.Mol)
        radius (int): the maximum radius of Morgan fingerprint

    Returns:
        list: a list of tuple (mol, a Morgan fingerprint for a molecule, the corresponding bit information)
    """
    fp_info_list = []
    for mol in mol_list:
        fp_info = calculate_mol_fp(mol, radius)
        fp_info_list.append((mol, fp_info))
    return fp_info_list


def get_bond_info(mol):
    """Collect the types of bonds in a molecule

    Args:
        mol (Mol):  rdkit.Chem.rdchem.Mol

    Returns:
        dict: bond type information {begin_atom_idx: {end_atom_idx: bond_type}}
                for example, {0: {1: 'DOUBLE'},
                            1: {0: 'DOUBLE', 2: 'SINGLE'},
                            2: {1: 'SINGLE', 3: 'AROMATIC', 19: 'AROMATIC'}, ... }
    """

    bond_info = defaultdict(lambda: defaultdict(str))
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        bond_info[begin_atom][end_atom] = str(bond.GetBondType())
        bond_info[end_atom][begin_atom] = str(bond.GetBondType())

    return bond_info


def check_bond(mol, bond_info, common_atom_idx, no_double=False, no_triple=False):
    """Check type of bond between substruture and non-substructure

    Args:
        mol (Mol): rdkit.Chem.rdchem.Mol
        bond_info (Dict): bond type information from get_bond_info()
        common_atom_idx (set()): the index set of atoms covered by common fingerprints
        no_double (bool, optional): remove atom with double bond. Defaults to False.
        no_triple (bool, optional): remove atom with triple bond. Defaults to False.

    Returns:
        int: the index of atom that should be excluded from substructure
    """

    for c_id in common_atom_idx:
        # common sub atom
        c_atom = mol.GetAtomWithIdx(c_id)
        c_atom_frag_neighbor_ids = [
            x.GetIdx() for x in c_atom.GetNeighbors() if x.GetIdx() not in common_atom_idx]
        c_atom_sub_neighbor_ids = [
            x.GetIdx() for x in c_atom.GetNeighbors() if x.GetIdx() in common_atom_idx]

        # TODO: consider more than one connected frag atoms
        if len(c_atom_frag_neighbor_ids) > 1:
            return c_id

        # stereo: chiral
        if c_atom_frag_neighbor_ids:
            if str(c_atom.GetChiralTag()) != 'CHI_UNSPECIFIED':
                return c_id

        c_atom_to_frag_bond_type = set(
            [bond_info[c_id][j] for j in c_atom_frag_neighbor_ids])
        # c_atom_to_frag_bond_type = set([str(mol.GetBondBetweenAtoms(c_id, j).GetBondType()) for j in c_atom_frag_neighbor_ids])

        if 'AROMATIC' in c_atom_to_frag_bond_type:
            return c_id
        if no_double and 'DOUBLE' in c_atom_to_frag_bond_type:
            return c_id
        if no_triple and 'TRIPLE' in c_atom_to_frag_bond_type:
            return c_id

        # double bond could not be broken at aromatic atoms
        if 'DOUBLE' in c_atom_to_frag_bond_type and c_atom.GetIsAromatic():
            return c_id

        for frag_neighbor_id in c_atom_frag_neighbor_ids:
            # stereo: chiral
            if str(mol.GetAtomWithIdx(frag_neighbor_id).GetChiralTag()) != 'CHI_UNSPECIFIED':
                return c_id

            # stereo: double bond
            c2f_bond = mol.GetBondBetweenAtoms(c_id, frag_neighbor_id)
            if str(c2f_bond.GetStereo()) != 'STEREONONE':
                return c_id

            # stereo: double bond, check 2-step neighbors
            for frag_neighbor_atom2 in mol.GetAtomWithIdx(frag_neighbor_id).GetNeighbors():
                frag_neighbor_id2 = frag_neighbor_atom2.GetIdx()
                if frag_neighbor_id2 not in common_atom_idx:
                    f2f_bond = mol.GetBondBetweenAtoms(
                        frag_neighbor_id2, frag_neighbor_id)
                    if str(f2f_bond.GetStereo()) != 'STEREONONE':
                        # stereo_atom_ids = set(f2f_bond.GetStereoAtoms())
                        # if c_id in stereo_atom_ids:
                        #     return c_id
                        return c_id

        # stereo: double bond, check 2-step neighbors
        for sub_neighbor_id in c_atom_sub_neighbor_ids:
            c2sub_bond = mol.GetBondBetweenAtoms(c_id, sub_neighbor_id)
            if str(c2sub_bond.GetStereo()) != 'STEREONONE':
                # stereo_atom_ids = set(c2sub_bond.GetStereoAtoms())
                # for frag_neighbor_id in c_atom_frag_neighbor_ids:
                #     if frag_neighbor_id in stereo_atom_ids:
                #         return c_id
                return c_id
    return None

def get_atom_idx_of_radiusN(mol, atom_env):
    if type(atom_env) is not list:
        atom_env = [atom_env]

    atom_idx_set = set()
    for item in atom_env:
        cur_idx = {}
        cur_atom, cur_radius = item[0], item[1]
        cur_env = Chem.FindAtomEnvironmentOfRadiusN(mol, cur_radius, cur_atom)
        Chem.PathToSubmol(mol, cur_env, atomMap=cur_idx)
        atom_idx_set.update(set(cur_idx.keys()))
    return atom_idx_set


def get_submol_in_env(mol, atom_env: list):
    """Get submol with given atom enviroments

    Args:
        mol (Mol):  rdkit.Chem.rdchem.Mol
        atom_env (list): list of atom env [(center_atom, radius)]
    Returns:
        tuple: (submol, the corresponding atom index in origin mol)
    """
    atom_idx_set = get_atom_idx_of_radiusN(mol, atom_env)

    # TODO: fix bond breaking?
    mol_rw = Chem.RWMol(mol)
    mol_idx = list(range(len(mol.GetAtoms())))
    for i in atom_idx_set:
        mol_idx.remove(i)
    mol_idx.reverse()

    for i in mol_idx:
        mol_rw.RemoveAtom(i)

    return mol_rw, atom_idx_set


def get_similarity(mol_1, mol_2, radius=2):
    fp_1 = AllChem.GetMorganFingerprintAsBitVect(mol=mol_1,
                                                 radius=radius,
                                                 useChirality=True)
    fp_2 = AllChem.GetMorganFingerprintAsBitVect(mol=mol_2,
                                                 radius=radius,
                                                 useChirality=True)

    return DataStructs.TanimotoSimilarity(fp_1, fp_2)


def need_to_add_Hs(atom):
    """Check if current atom need to add H when breaking the bond connected to this atom

    Args:
        atom (Atom): Atom

    Returns:
        Boolean: True if need to add H 
    """
    return atom.GetFormalCharge() != 0 or \
        atom.GetIsAromatic() or atom.GetSymbol() in [
        'Si', 'P', 'S', 'Ta', 'Sn', 'As', 'Se', 'Te']


def split_mol(mol, atom_idx):
    """Split molecule into substructure and fragments

    Args:
        mol (Mol): input molecule
        atom_idx (set(int)): atoms ids of the substructure

    Returns:
        Tuple: isotope labeled sub and frag mol
    """
    sub_mol, frag_mol = Chem.RWMol(mol), Chem.RWMol(mol)

    atom_num = 0
    for _ in mol.GetAtoms():
        atom_num += 1

    for idx in range(atom_num - 1, -1, -1):
        if idx in atom_idx:
            # atom_to_remove from frag mol
            atom_to_remove = frag_mol.GetAtomWithIdx(idx)
            atom_id2num_add_H = {}

            for neighbor_atom in atom_to_remove.GetNeighbors():
                neighbor_atom_id = neighbor_atom.GetIdx()
                # the removed atom has neighbor in frag mol
                if neighbor_atom_id not in atom_idx and need_to_add_Hs(frag_mol.GetAtomWithIdx(neighbor_atom_id)):
                    bond_type = frag_mol.GetBondBetweenAtoms(
                        neighbor_atom_id, idx).GetBondType()
                    if bond_type == Chem.BondType.SINGLE:
                        addH = 1
                    elif bond_type == Chem.BondType.DOUBLE:
                        addH = 2
                    else:
                        assert bond_type == Chem.BondType.TRIPLE
                        addH = 3
                    atom_id2num_add_H[neighbor_atom_id] = addH

            for atom_id_to_add_H, num_H in atom_id2num_add_H.items():
                for _ in range(num_H):
                    new_add_H_id = frag_mol.AddAtom(Chem.Atom(1))
                    frag_mol.AddBond(atom_id_to_add_H,
                                     new_add_H_id, Chem.BondType.SINGLE)
        else:
            # atom_to_remove from frag mol
            atom_to_remove = sub_mol.GetAtomWithIdx(idx)
            atom_id2num_add_H = {}
            for neighbor_atom in atom_to_remove.GetNeighbors():
                neighbor_atom_id = neighbor_atom.GetIdx()
                # the removed atom has neighbor in sub mol
                if neighbor_atom_id in atom_idx and need_to_add_Hs(sub_mol.GetAtomWithIdx(neighbor_atom_id)):
                    bond_type = sub_mol.GetBondBetweenAtoms(
                        neighbor_atom_id, idx).GetBondType()
                    if bond_type == Chem.BondType.SINGLE:
                        addH = 1
                    elif bond_type == Chem.BondType.DOUBLE:
                        addH = 2
                    else:
                        assert bond_type == Chem.BondType.TRIPLE
                        addH = 3
                    atom_id2num_add_H[neighbor_atom_id] = addH

            for atom_id_to_add_H, num_H in atom_id2num_add_H.items():
                for _ in range(num_H):
                    new_add_H_id = sub_mol.AddAtom(Chem.Atom(1))
                    sub_mol.AddBond(atom_id_to_add_H,
                                    new_add_H_id, Chem.BondType.SINGLE)

    for idx in range(atom_num - 1, -1, -1):
        if idx in atom_idx:
            frag_mol.RemoveAtom(idx)
        else:
            sub_mol.RemoveAtom(idx)

    # Hack to fix when "HasSubstructMatch" in label_query_mol might fail
    # the reason might be that the submol has [H], don't know why rdkit fails
    sub_mol = Chem.MolFromSmiles(Chem.MolToCXSmiles(sub_mol))
    frag_mol = Chem.MolFromSmiles(Chem.MolToCXSmiles(frag_mol))
    return sub_mol, frag_mol


def remove_isotope(input_mol):
    output_mol = Chem.RWMol(input_mol)
    for atom in output_mol.GetAtoms():
        if atom.GetIsotope() < 100:
            atom.SetIsotope(0)

    for new_atom, old_atom in zip(output_mol.GetAtoms(), input_mol.GetAtoms()):
        assert new_atom.GetAtomicNum() == old_atom.GetAtomicNum()

    return output_mol


def add_fp_prop(input_mol, fp_info=None):
    if fp_info is None:
        fp_info = calculate_mol_fp(input_mol, 2)
    for fp, atom_id_fpradius_list in fp_info.items():
        for (atom_id, fpradius) in atom_id_fpradius_list:
            if fpradius in [0, 1, 2]:
                input_mol.GetAtomWithIdx(atom_id).SetProp(
                    f'{fingerprint_prop_key}_{fpradius}', str(fp))
    return input_mol


def remove_fp_prop(input_mol):
    for atom in input_mol.GetAtoms():
        for fpradius in [0, 1, 2]:
            p_name = f'{fingerprint_prop_key}_{fpradius}'
            if atom.HasProp(p_name):
                atom.ClearProp(p_name)
    return input_mol


def add_bond(mol, begin_atom, end_atom, bond_type_str, removed_atom_ids):
    """Add bond between two atoms

    Args:
        mol (rdkit.Chem.rdchem.RwMol): RwMol of substructure and fragment
        begin_atom (rdkit.Chem.rdchem.Atom): atom in substructure
        end_atom (rdkit.Chem.rdchem.Atom): atom in fragment
        bond_type_str (str): 'SINGLE', 'DOUBLE', 'TRIPLE'
        removed_atom_ids (set): set of H atom ids to remove further
    Return:
        mol (rdkit.Chem.rdchem.RwMol): mol with bond added
    """
    if bond_type_str == 'SINGLE':
        bond_type = Chem.BondType.SINGLE
        removeH = 1
    elif bond_type_str == 'DOUBLE':
        bond_type = Chem.BondType.DOUBLE
        removeH = 2
    else:
        assert bond_type_str == 'TRIPLE'
        bond_type = Chem.BondType.TRIPLE
        removeH = 3

    mol.AddBond(begin_atom.GetIdx(), end_atom.GetIdx(), bond_type)

    if removed_atom_ids is None:
        return mol

    b_num = 0
    for begin_atom_neighbor_atom in begin_atom.GetNeighbors():
        if begin_atom_neighbor_atom.GetIdx() in removed_atom_ids:
            continue
        if begin_atom_neighbor_atom.GetAtomicNum() == 1:
            removed_atom_ids.append(begin_atom_neighbor_atom.GetIdx())
            b_num += 1
            if b_num == removeH:
                break
    e_num = 0
    for end_atom_neighbor_atom in end_atom.GetNeighbors():
        if end_atom_neighbor_atom.GetIdx() in removed_atom_ids:
            continue
        if end_atom_neighbor_atom.GetAtomicNum() == 1:
            removed_atom_ids.append(end_atom_neighbor_atom.GetIdx())
            e_num += 1
            if e_num == removeH:
                break

    return mol


SUB = 20  # Label of atom in substructures
FRAG = 40  # Label of atom in fragments


def merge_with_prop(mols, is_sanitized):
    """Merge substructures and fragments

    Args:
        mols (tuple of rdkit.Chem.rdchem.Mol): (substructure, fragment)
        is_sanitized (bool): whether the mol is sanitized
    Return:
        mol_rw (rdkit.Chem.rdchem.RwMol): Merged mol
    """
    sub_mol, frag_mol = mols
    mol_rw = Chem.RWMol(sub_mol)
    isotopic_num = set()

    for atom in mol_rw.GetAtoms():
        _label = atom.GetIsotope()
        if _label > 0:
            isotopic_num.add(_label)
            atom.SetIsotope(_label + SUB)

    # Insert mol of fragments into substructure mol
    if frag_mol:
        mol_rw.InsertMol(frag_mol)
        for atom in mol_rw.GetAtoms():
            _label = atom.GetIsotope()
            if 0 < _label < SUB:
                isotopic_num.add(_label)
                atom.SetIsotope(_label + FRAG)
    if is_sanitized:
        mol_rw = Chem.RWMol(Chem.AddHs(mol_rw))
        extra_h_atom_ids = []
    else:
        extra_h_atom_ids = None
    while isotopic_num:
        cur_isotopic_num = isotopic_num.pop()
        atom_in_sub, atom_in_frag = [], []
        for atom in mol_rw.GetAtoms():
            if atom.GetIsotope() == (cur_isotopic_num + SUB):
                atom_in_sub.append(atom)
            if atom.GetIsotope() == (cur_isotopic_num + FRAG):
                atom_in_frag.append(atom)

        if len(atom_in_frag) == 0 or len(atom_in_sub) == 0:
            continue

        if len(atom_in_frag) == 1:
            for _atom in atom_in_sub:
                begin_atom = _atom
                end_atom = atom_in_frag[0]

                bond_type_str = begin_atom.GetProp('bond_type')
                mol_rw = add_bond(mol_rw, begin_atom, end_atom,
                                  bond_type_str, extra_h_atom_ids)

        elif len(atom_in_sub) == 1:

            for _atom in atom_in_frag:
                begin_atom = atom_in_sub[0]
                end_atom = _atom

                bond_type_str = begin_atom.GetProp('bond_type')
                mol_rw = add_bond(mol_rw, begin_atom, end_atom,
                                  bond_type_str, extra_h_atom_ids)

        else:
            return '[Error]'

    if extra_h_atom_ids:
        extra_h_atom_ids.sort(reverse=True)
        for r_id in extra_h_atom_ids:
            mol_rw.RemoveAtom(r_id)

    for atom in mol_rw.GetAtoms():
        if atom.GetIsotope() < 100:
            atom.SetIsotope(0)
    return Chem.RemoveHs(mol_rw, implicitOnly=True)


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def mol_prop_to_note(mol, pname='atomNote_0'):
    mol_rw = Chem.RWMol(mol)
    for atom in mol_rw.GetAtoms():
        if atom.HasProp(pname):
            atom.SetProp('atomNote', atom.GetProp(pname))
    return mol_rw


def remove_small_pieces(mol, common_atom_ids, min_atoms=4):
    if not common_atom_ids:
        return common_atom_ids
    mol_rw = Chem.RWMol(mol)
    mol_rw = mol_with_atom_index(mol_rw)
    sub_mol, _ = split_mol(mol_rw, common_atom_ids)
    for smi_piece in Chem.MolToSmiles(sub_mol).split('.'):
        mol_piece = Chem.MolFromSmiles(smi_piece)
        if mol_piece and mol_piece.GetNumHeavyAtoms() < min_atoms:
            for atom_in_piece in mol_piece.GetAtoms():
                common_atom_ids.remove(atom_in_piece.GetAtomMapNum())
    return common_atom_ids


def test_merge_sub_frag(labeled_target_sub, labeled_target_frag_smi, t):
    """merge substructure with fragments

    Args:
        labeled_target_sub (Mol): substructure mol
        labeled_target_frag_smi (str): predicted frag SMILES
        t (str): SMILES of target

    Returns:
        Tuple(bool, str): First => (True: correct, False: incorrect, None: merge failed), 
                          Second => SMILES of merged molecule
    """

    if labeled_target_frag_smi is None:
        labeled_target_frag_smi = ''
    frag_mol = Chem.MolFromSmiles(
        labeled_target_frag_smi, sanitize=True)
    is_sanitized = True

    if labeled_target_frag_smi and frag_mol is None:
        is_sanitized = False
        frag_mol = Chem.MolFromSmiles(labeled_target_frag_smi, sanitize=False)

    try:
        pred_t_mol = merge_with_prop(
            [labeled_target_sub, frag_mol], is_sanitized)
    except KeyError:
        # merge failed
        return None, '[Error]'
    except AssertionError:
        # merge failed
        return None, '[Error]'
    if pred_t_mol == '[Error]':
        return None, '[Error]'

    pred_t_smi = Chem.MolToSmiles(pred_t_mol)
    tgt_smi = canonicalize_smiles(t)
    if not is_sanitized:
        pred_t_smi = postprocess(pred_t_smi)

    pred_t_smi_ = canonicalize_smiles(pred_t_smi)
    if not pred_t_smi_:
        return None, '[Error]'
    if tgt_smi == pred_t_smi_:
        return True, pred_t_smi_

    return False, pred_t_smi_


def remove_atom_mapping(mol):
    """remove mapping atom index

    Args:
        mol (Mol):  Mol object from rdkit.Chem.MolFromSmiles()

    Returns:
        Mol: mol without atom mapping information
    """

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol