import copy
import random
from collections import defaultdict

from rdkit import Chem

from utils.common_utils import PropNames

from .mol_utils import (add_fp_prop, check_bond, fingerprint_prop_key,
                        remove_isotope, remove_small_pieces, split_mol,
                        test_merge_sub_frag)


class AtomChecker(object):
    def __init__(self, query, pName):
        # identify the atoms that have the properties we care about
        self._atsToExamine = [(x.GetIdx(), x.GetProp(pName)) for x in query.GetAtoms()
                              if x.HasProp(pName)]
        self._pName = pName

    def __call__(self, mol, vect):
        if not self._atsToExamine:
            return 1
        matched_atom_count = 0
        for idx, qtyp in self._atsToExamine:
            midx = vect[idx]
            atom = mol.GetAtomWithIdx(midx)
            if atom.HasProp(self._pName) and atom.GetProp(self._pName) == qtyp:
                matched_atom_count += 1
        return matched_atom_count


def update_atom_idx(mol, bond_info, common_atom_idx):
    """Check type of bond between substruture and non-substructure and update the index of common_atom
    Args:
        mol (Mol):  rdkit.Chem.rdchem.Mol
        bond_info (dict): bond type information from get_bond_info()
        common_atom_idx (set()): the index set of atoms covered by common fingerprints

    Returns:
        set(): atoms ids of common substructure
    """

    pre_num_total_sub_atoms = len(common_atom_idx)
    cur_idx = copy.deepcopy(common_atom_idx)
    while True:
        atom_idx = check_bond(mol, bond_info, cur_idx)

        while atom_idx is not None:
            cur_idx.remove(atom_idx)
            atom_idx = check_bond(mol, bond_info, cur_idx)

        from .extract_utils import label_mol
        atom_idx = label_mol(mol, cur_idx, True)
        while atom_idx is not None:
            cur_idx.remove(atom_idx)
            atom_idx = label_mol(mol, cur_idx, True)

        cur_num_total_sub_atoms = len(cur_idx)
        if pre_num_total_sub_atoms == cur_num_total_sub_atoms:
            break
        pre_num_total_sub_atoms = cur_num_total_sub_atoms

    cur_idx = remove_small_pieces(mol, cur_idx)
    return cur_idx


def label_mol(mol, atom_idx, is_remove_multi2multi=False):
    """Add isotope label to atom in substructures

    Args:
        mol (Mol): input molecule
        atom_idx (set(int)): atom ids of substructure
        is_remove_multi2multi (bool, optional): if true, we do not add isotope, and this method will help
        obtain atoms that would be labeled with multiple isotope numbers. Defaults to False.

    Returns:
        labeled molecule
    """

    mol_rw = Chem.RWMol(mol)
    break_bond_atom = defaultdict(set)

    for i in atom_idx:
        atom = mol_rw.GetAtomWithIdx(i)
        for x in atom.GetNeighbors():
            _idx = x.GetIdx()
            if _idx not in atom_idx:
                break_bond_atom[i].add(_idx)

    isotopic_mark = 1
    marked_atoms = {}
    act_atoms = copy.deepcopy(break_bond_atom)

    while act_atoms:
        sub_atom_idx, frag_atom_idx = act_atoms.popitem()

        for cur_atom_idx in frag_atom_idx:
            sub_atom = mol_rw.GetAtomWithIdx(sub_atom_idx)
            frag_atom = mol_rw.GetAtomWithIdx(cur_atom_idx)

            cur_bond_type = str(mol_rw.GetBondBetweenAtoms(
                sub_atom_idx, cur_atom_idx).GetBondType())
            cur_chiral_tag = str(sub_atom.GetChiralTag())

            if cur_atom_idx not in marked_atoms:
                frag_atom.SetIsotope(isotopic_mark)
                sub_atom.SetIsotope(isotopic_mark)

                sub_atom.SetProp(PropNames.Bond_Type, cur_bond_type)
                sub_atom.SetProp(PropNames.Chiral_Tag, cur_chiral_tag)

                if is_remove_multi2multi and sub_atom_idx in marked_atoms:
                    # sub_atom_idx already has isotope number
                    # this is mainly because the atom has multiple neighbor atoms in different fragments
                    # for simplicity, we remove these atoms from substructure.
                    return sub_atom_idx

                assert sub_atom_idx not in marked_atoms
                marked_atoms[cur_atom_idx] = isotopic_mark
                marked_atoms[sub_atom_idx] = isotopic_mark
                isotopic_mark += 1
            else:
                cur_mark = marked_atoms[cur_atom_idx]
                sub_atom.SetIsotope(cur_mark)
                sub_atom.SetProp(PropNames.Bond_Type, cur_bond_type)
                sub_atom.SetProp(PropNames.Chiral_Tag, cur_chiral_tag)

                if is_remove_multi2multi and sub_atom_idx in marked_atoms:
                    # sub_atom_idx already has isotope number
                    # this is mainly because the atom has multiple neighbor atoms in different fragments
                    # for simplicity, we remove these atoms from substructure.
                    return sub_atom_idx

                assert sub_atom_idx not in marked_atoms
                marked_atoms[sub_atom_idx] = cur_mark

    if is_remove_multi2multi:
        # great, no atoms will be removed from substructure
        return None
    return atom_idx, mol_rw


def get_sub_mol(mol, atom_idx):
    """Get substruture and fragments with isotopic label

    Args:
        mol (Mol): rdkit.Chem.rdchem.Mol of the query molecule
        atom_idx (dict): atom index

    Returns:
        tuple: (substruture, substruture with isotopic label, fragments with isotopic label)            
    """
    sub_atom_idx, labeled_mol = label_mol(mol, atom_idx)
    labeled_sub, labeled_frag = split_mol(labeled_mol, sub_atom_idx)
    return remove_isotope(labeled_sub), labeled_sub, labeled_frag, labeled_mol


def label_query_mol(query_mol, labeled_sub_mol, sub_mol, sub_src_mol, allow_add_isotope, num_try=10):
    """add isotope to the query mol, and the isotope number should be same with the given sub

    Args:
        query_mol (Mol): the query mol
        labeled_sub_mol (Mol): the labeled sub mol
        sub_mol (Mol): sub mol with isotope removed
        sub_src_mol (Mol): the souce where the sub mol is extracted, mainly for debug purpose
        allow_add_isotope (bool): whether new isotope number is allowed to add
        num_try (int, optional): try different randomized SMILES during substructure match, we might failed to 
        get the correct substructure mapping using GetSubstructMatches when part of the substructure is symmetrical.
        We try multiple times with randomized SMILES to mitigate this. Defaults to 10.

    Returns:
        Tuple: istope labeled sub, istope labeled frag, and istope labeled query_mol
    """
    # TODO: try better solutions to add isotope labels
    for radius in [2, 1, 0]:
        # when do substructure match, we add additional fingerprint constraints
        # following https://www.rdkit.org/docs/GettingStartedInPython.html#advanced-substructure-matching
        # we use fp radius in decreasing order from 2 to 0
        for _ in range(num_try):
            trial_result = label_query_mol_trial(
                query_mol, labeled_sub_mol, sub_mol, radius, sub_src_mol, allow_add_isotope)
            if trial_result:
                return trial_result
    #Draw.MolsToImage([labeled_sub_mol, sub_src_mol, query_mol], highlightAtomLists=[None, sub_src_mol.GetSubstructMatches(sub_mol)[0], query_mol.GetSubstructMatches(sub_mol)[0]], subImgSize=(400,400), legends=['sub','sub_src', 'target']).save('debug.png')
    return None


def label_query_mol_trial(query_mol, labeled_sub_mol, sub_mol, radius, sub_src_mol, allow_add_isotope):
    split_query_mol = [Chem.MolFromSmiles(
        smiles) for smiles in Chem.MolToSmiles(query_mol).split(".")]
    random.shuffle(split_query_mol)
    query_mol = Chem.MolFromSmiles(
        ".".join([Chem.MolToSmiles(mol, doRandom=True) for mol in split_query_mol]))
    query_mol = add_fp_prop(query_mol)

    # additional constraint to get matched substructure
    # https://www.rdkit.org/docs/GettingStartedInPython.html#advanced-substructure-matching
    params = Chem.SubstructMatchParameters()
    checker = AtomChecker(sub_mol, f'{fingerprint_prop_key}_{radius}')
    params.setExtraFinalCheck(checker)
    params.useChirality = True
    matches = query_mol.GetSubstructMatches(sub_mol, params)

    possible_reasonable_results = []

    for query_match_atom_idx in matches:
        mol_rw = Chem.RWMol(query_mol)
        fp_match_count = checker(query_mol, query_match_atom_idx)

        max_isotope_label_sub = 0
        assert_atom_id = 0
        for atom in labeled_sub_mol.GetAtoms():
            assert atom.GetIdx() == assert_atom_id
            assert_atom_id += 1
            if atom.GetIsotope() != 0:
                max_isotope_label_sub = max(
                    max_isotope_label_sub, atom.GetIsotope())
                _idx = query_match_atom_idx[atom.GetIdx()]
                _label = atom.GetIsotope()
                _query_atom = mol_rw.GetAtomWithIdx(_idx)

                _query_atom.SetIsotope(_label)
                cur_chiral_tag = str(_query_atom.GetChiralTag())
                _query_atom.SetProp(PropNames.Chiral_Tag, cur_chiral_tag)

                for x in _query_atom.GetNeighbors():
                    if x.GetIdx() not in query_match_atom_idx:
                        x.SetIsotope(_label)
                        cur_bond_type = str(mol_rw.GetBondBetweenAtoms(
                            _query_atom.GetIdx(), x.GetIdx()).GetBondType())
                        _query_atom.SetProp(PropNames.Bond_Type, cur_bond_type)

        if allow_add_isotope:
            max_isotope_label_sub += 1
            assert_atom_id = 0
            for atom in labeled_sub_mol.GetAtoms():
                assert atom.GetIdx() == assert_atom_id
                assert_atom_id += 1
                if atom.GetIsotope() == 0:
                    _idx = query_match_atom_idx[atom.GetIdx()]
                    _query_atom = mol_rw.GetAtomWithIdx(_idx)
                    has_fag_neighbors = False
                    for x in _query_atom.GetNeighbors():
                        # has_fag_neighbors??
                        # TODO: do we need to check if the atoms in frag are labeled already?
                        if x.GetIdx() not in query_match_atom_idx:
                            x.SetIsotope(max_isotope_label_sub)
                            cur_bond_type = str(mol_rw.GetBondBetweenAtoms(
                                _query_atom.GetIdx(), x.GetIdx()).GetBondType())
                            _query_atom.SetProp(
                                PropNames.Bond_Type, cur_bond_type)
                            has_fag_neighbors = True
                    if has_fag_neighbors:
                        _query_atom.SetIsotope(max_isotope_label_sub)
                        cur_chiral_tag = str(_query_atom.GetChiralTag())
                        _query_atom.SetProp(
                            PropNames.Chiral_Tag, cur_chiral_tag)
                        max_isotope_label_sub += 1
        try:
            labeled_query_sub_mol, labeled_frag = split_mol(
                mol_rw, set(query_match_atom_idx))
        except AssertionError:
            # split mol do not support aromatic
            continue

        merge_flag, _ = test_merge_sub_frag(labeled_query_sub_mol, Chem.MolToSmiles(
            labeled_frag), Chem.MolToSmiles(query_mol))
        if merge_flag:
            if not labeled_query_sub_mol.HasSubstructMatch(sub_mol, useChirality=True) or not sub_mol.HasSubstructMatch(remove_isotope(labeled_query_sub_mol), useChirality=True):
                # assert labeled_query_sub_mol.HasSubstructMatch(sub_mol, useChirality=True)
                # Draw.MolsToImage([labeled_sub_mol, labeled_query_sub_mol, sub_src_mol, mol_rw], molsPerRow=2, subImgSize=(400,400), legends=['input_sub','output_sub','input_sub_src','target']).save('debug.png')
                pass
            else:
                possible_reasonable_results.append(
                    (fp_match_count, labeled_query_sub_mol, labeled_frag, mol_rw))

    possible_reasonable_results.sort(key=lambda item: item[0], reverse=True)
    if possible_reasonable_results:
        # if len(possible_reasonable_results)>1 and possible_reasonable_results[0][0]!=possible_reasonable_results[1][0]:
        #     Draw.MolsToImage([sub_mol, possible_reasonable_results[0][1], possible_reasonable_results[1][1], query_mol], subImgSize=(400,400), legends=['input_sub','sub_1','sub_2', 'target']).save('debug.png')
        #     print('investigate')
        return possible_reasonable_results[0][1], possible_reasonable_results[0][2], possible_reasonable_results[0][3]
    return None


def resplit(tgt_mol, sub_mol, labeled_sub_mol, sub_src_mol=None):
    """resplit the target mol given the sub_mol and sub mol with isotope label

    Args:
        tgt_mol (Mol): target mol
        sub_mol (Mol): substructure mol
        labeled_sub_mol (Mol): substructure mol with isotope label
        sub_src_mol (Mol, optional): molecule from which substructure is extracted, for debug purpose. Defaults to None.

    Returns:
        Tuple: istope labeled sub (from target), istope labeled frag (from target), and istope labeled target
    """
    if not tgt_mol.HasSubstructMatch(sub_mol, useChirality=True):
        # Draw.MolsToGridImage([tgt_mol, sub_mol, sub_src_mol], highlightAtomLists = [None, None, sub_src_mol.GetSubstructMatches(sub_mol)[0]],subImgSize=(400,400), legends=['tgt_mol','sub_mol','sub_src_mol']).save('debug.png')
        return None
    return label_query_mol(query_mol=tgt_mol, labeled_sub_mol=labeled_sub_mol, sub_mol=sub_mol, sub_src_mol=sub_src_mol, allow_add_isotope=False)
