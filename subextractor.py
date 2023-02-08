#!/usr/bin/env python
# coding: utf-8

from collections import Counter
from itertools import combinations

from func_timeout import func_set_timeout
from rdkit import Chem

from utils.extract_utils import get_sub_mol, label_query_mol, update_atom_idx
from utils.mol_utils import *

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)



class CollectAtomIndex():
    """Collect atom index to build substructures based on fingerprint
    """

    def __init__(self,
                 query,
                 target,
                 candidates,
                 max_fp_radius=6,
                 min_count=5,
                 min_fp_radius=2,
                 reactions=None):
        """Collect the atom index of reaction-aware substructure with

        Args:
            query (Mol): rdkit.Chem.rdchem.Mol of the query molecule
            target (Mol): rdkit.Chem.rdchem.Mol of the target molecule
            candidates (Mol): List<rdkit.Chem.rdchem.Mol> of the candidate molecules
            max_fp_radius (int, optional): the maximum radius of Morgan fingerprints. Defaults to 6.
            min_count (int, optional): the min count of common fingerprints . Defaults to 5.
            min_fp_radius (int, optional): the min radius of Morgan fingerprints. Defaults to 2.            
        """

        self.max_fp_radius = max_fp_radius
        self.min_count = min_count
        self.min_fp_radius = min_fp_radius
        self.reactions = reactions

        self.query = query
        self.target = target
        self.candidates = candidates

        self.query_fp_info = calculate_mol_fp(mol=query,
                                                 radius=max_fp_radius)

        self.target_fp_info = calculate_mol_fp(mol=target,
                                                  radius=max_fp_radius,)

        self.candidate_fps_info = calculate_mols_fps(mol_list=candidates,
                                                     radius=max_fp_radius,
                                                     )

    def get_common_fps(self):
        """Get common fingerprints between query and candidates

        Returns:
            tuple(dict, list): 
                    - dict: common fp and enviroment information in different candidates
                      for example: {<rdkit.Chem.rdchem.Mol at 0x25d94bafbe0>: [(279194284, ((11, 6),)),
                                                                             (488798576, ((12, 6),)),
                                                                             (837524623, ((8, 6),)),
                                                                             (2011874252, ((13, 6),)), ...], ...}
                    - list: common fps and count
                      for example: [(1894212005, 20), (3866113095, 20), ...]
        """

        common_fp_in_candidates = {}
        common_fp = []

        for mol, info in self.candidate_fps_info:
            fp = {}
            query_fp_list = list(self.query_fp_info.keys())
            for item in info.items():
                if item[0] in query_fp_list and item[1][0][1] >= self.min_fp_radius:
                    fp[item[0]] = item[1]

            fp_list = sorted(
                fp.items(), key=lambda x: x[1][0][1], reverse=True)

            common_fp_in_candidates[mol] = fp_list
            common_fp.extend([f[0] for f in fp_list])

        count_common_fp = sorted(
            Counter(common_fp).items(), key=lambda x: x[-1], reverse=True)

        filtered_fp = []
        if count_common_fp:
            i = 0
            max_num = count_common_fp[i][1]
            while max_num >= self.min_count and i < len(count_common_fp) - 1:
                filtered_fp.append(count_common_fp[i])
                i += 1
                max_num = count_common_fp[i][1]

        return common_fp_in_candidates, filtered_fp

    
    def get_target_atom_idx(self):
        """Get index of the atoms covered by common fingerprints of target

        Returns:
            set(): the atom index covered by common fingerprints in target
        """
        # TODO: optimize the extraction process, avoid re-computing
        _, filtered_fp = self.get_common_fps()
        common_fp = [i[0] for i in filtered_fp]
        target_fp = [i for i in self.target_fp_info.keys()]        
        fp_list = list(set(target_fp) & set(common_fp))
        tgt_smi = Chem.MolToSmiles(self.target)
        if tgt_smi in self.reactions:
            for pseudo_src_smi in self.reactions[tgt_smi]:
                pseudo_src_mol = Chem.MolFromSmiles(pseudo_src_smi)
                if get_similarity(pseudo_src_mol, self.query)<0.2:
                    return {}
                pseudo_src_fp_info = calculate_mol_fp(pseudo_src_mol,self.max_fp_radius)
                pseudo_src_fp_list = [i for i in pseudo_src_fp_info.keys()]
                fp_list = list(set(fp_list) & set(pseudo_src_fp_list))
        target_atom_idx = set()
        env_list = []
        for fp in fp_list:
            env_list.extend([m for m in self.target_fp_info[fp]])

        _, target_atom_idx = get_submol_in_env(
            mol=self.target, atom_env=env_list)
        tgt_bond_info = get_bond_info(self.target)
        updated_target_atom_idx = update_atom_idx(mol=self.target,
                                                                   bond_info=tgt_bond_info,
                                                                   common_atom_idx=target_atom_idx)

        return updated_target_atom_idx

    def get_query_atom_idx(self):
        # TODO: optimize the extraction process, avoid re-computing
        _, filtered_fp = self.get_common_fps()
        common_fp = [i[0] for i in filtered_fp]
        query_fp = [i for i in self.query_fp_info.keys()]
        fp_list = list(set(query_fp) & set(common_fp))
        query_atom_idx = set()
        env_list = []
        for fp in fp_list:
            env_list.extend([m for m in self.query_fp_info[fp]])
        _, query_atom_idx = get_submol_in_env(
            mol=self.query, atom_env=env_list)
        query_bond_info = get_bond_info(self.query)
        updated_query_bond_info_atom_idx = update_atom_idx(mol=self.query,
                                                                            bond_info=query_bond_info,
                                                                            common_atom_idx=query_atom_idx)

        return updated_query_bond_info_atom_idx


class SubMolExtractor():
    """
    extract substructure and fragments
    """
    def __init__(self,
                 query,
                 target,
                 candidates,
                 reactions,
                 max_fp_radius=6,
                 min_count=5,
                 min_fp_radius=2):
        """init extractor

        Args:
            query (Mol): rdkit.Chem.rdchem.Mol of the query molecule
            target (Mol): rdkit.Chem.rdchem.Mol of the target molecule
            candidates (List<Mol>): List<rdkit.Chem.rdchem.Mol> of the candidate molecules
            reactions (Dict): reactions of retrieved candidates on train and dev set
            max_fp_radius (int, optional): the maximum radius of Morgan fingerprints. Defaults to 6.
            min_count (int, optional): the min count common fingerprints. Defaults to 5.
            min_fp_radius (int, optional): the min radius of Morgan fingerprints. Defaults to 2.
        """

        self.collect_atom_idx = CollectAtomIndex(query, target, candidates, max_fp_radius, min_count, min_fp_radius,
                                                 reactions)
        self.query = query
        self.target = target
        self.candidates = candidates
        self.reactions = reactions
        self.max_fp_radius = max_fp_radius
        self.query_fp_info = self.collect_atom_idx.query_fp_info
        self.target_fp_info = self.collect_atom_idx.target_fp_info
        self.query_atom_idx = self.collect_atom_idx.get_query_atom_idx()
        self.target_atom_idx = self.collect_atom_idx.get_target_atom_idx()
    

    def filter_atom_idx(self, mol, fp_info):
        """one common fingerprint might have different number of environments in the query and candidate,
         we try to filter some environments, and select the common environments which contains by both 
         the query and candidate mol.

        Args:
            mol (Mol): candidate mol
            fp_info (Dict): fingerprint info of candidate mol

        Returns:
            set(int): atom ids of the common substructure
        """
        _, filtered_fp = self.collect_atom_idx.get_common_fps()
        mol_fp = [i for i in fp_info.keys()]
        common_fp = set(mol_fp) & set([i[0] for i in filtered_fp])

        common_mol_fp_info = {}
        invalid_mol_fp_info = {}
        invalid_query_fp_info = {}
        mol_env = []
        query_env = []

        for fp in common_fp:

            cur_fp_set = set()
            for item in fp_info[fp]:
                cur_fp_set.add(item)
            common_mol_fp_info[fp] = cur_fp_set

            if len(fp_info[fp]) != len(self.query_fp_info[fp]):
                invalid_mol_fp_info[fp] = [i for i in fp_info[fp]]
                invalid_query_fp_info[fp] = [i for i in self.query_fp_info[fp]]
            else:
                mol_env.extend([i for i in fp_info[fp]])
                query_env.extend([i for i in self.query_fp_info[fp]])

        # TODO: optimize the extraction process
        # the following could be significantly improved by 
        #   1: iterate the environment in the decreasing order of fp radius
        #   2: remove the environment covered by large FP radius.
        for fp, env_list in invalid_mol_fp_info.items():            
            if len(env_list) > len(self.query_fp_info[fp]):
                # chose from candidate
                for sub_env in combinations(env_list, len(self.query_fp_info[fp])):
                    cur_env_list = [i for i in sub_env]

                    mol_env.extend(cur_env_list)
                    query_env.extend([i for i in self.query_fp_info[fp]])

                    cur_tgt = get_submol_in_env(mol=mol, atom_env=mol_env)[0]
                    cur_src = get_submol_in_env(mol=self.query, atom_env=query_env)[0]
                    if Chem.MolToSmiles(cur_tgt) == Chem.MolToSmiles(cur_src):
                        break

                    for item in cur_env_list:
                        mol_env.remove(item)
                    for i in self.query_fp_info[fp]:
                        query_env.remove(i)
            else:
                # chose from query
                for _sub_env in combinations(self.query_fp_info[fp], len(env_list)):
                    _cur_env_list = [i for i in _sub_env]

                    mol_env.extend(env_list)
                    query_env.extend(_cur_env_list)

                    cur_tgt = get_submol_in_env(mol=mol, atom_env=mol_env)[0]
                    cur_src = get_submol_in_env(mol=self.query, atom_env=query_env)[0]
                    if Chem.MolToSmiles(cur_tgt) == Chem.MolToSmiles(cur_src):
                        break

                    for item in _cur_env_list:
                        query_env.remove(item)
                    for item in env_list:
                        mol_env.remove(item)

        _, update_mol_atom_idx = get_submol_in_env(mol=mol, atom_env=mol_env)
        mol_bond_info = get_bond_info(mol)
        mol_atom_idx = update_atom_idx(mol=mol,
                                                        bond_info=mol_bond_info,
                                                        common_atom_idx=update_mol_atom_idx)
        return mol_atom_idx
    
 
    @func_set_timeout(60)
    def extractor(self): 
        """
        the substructure extraction
        """       
        # target_mol is one of the candidates
        target_mol = self.target
        target_fp_info = self.target_fp_info
        target_atom_idx = self.target_atom_idx
        # get substructure based on fingerprint from target and all candidates
        target_sub, labeled_target_sub, labeled_target_frag, labeled_target = get_sub_mol(mol=target_mol,
                                                                                                    atom_idx=target_atom_idx)
        
        # get substructure based on fingerprint from query and all candidates
        query_sub, labeled_query_sub, labeled_query_frag, labeled_query = get_sub_mol(mol=self.query,
                                                                                      atom_idx=self.query_atom_idx)

        if canonicalize_smiles(Chem.MolToSmiles(labeled_query_sub)) == canonicalize_smiles(Chem.MolToSmiles(labeled_target_sub)):
            # substructure from query and target is same, done
            return labeled_query_sub, labeled_query_frag, labeled_query, labeled_target_sub, labeled_target_frag, labeled_target 

        elif self.query.HasSubstructMatch(target_sub, useChirality=True):
            # query has matched substructure of target_sub, use target_sub to resplit the query mol
            # we allow add isotope here, as the query might have atoms with neighbors not in the target_sub 
            extr_res = label_query_mol(query_mol=self.query, labeled_sub_mol=labeled_target_sub,
                                                                                    sub_mol=target_sub,
                                                                                    sub_src_mol=target_mol,
                                                                                    allow_add_isotope=True)
            if extr_res:
                labeled_query_sub_mol, labeled_query_frag, labeled_query = extr_res
                return labeled_query_sub_mol, labeled_query_frag, labeled_query, labeled_target_sub, labeled_target_frag, labeled_target
            else:
                return None
        else:
            # one common fingerprint might have different number of environments in the query and candidate,
            # we try to filter some environments, and select the common environments which contains by both 
            # the query and candidate mol.
            updated_target_atom_idx = self.filter_atom_idx(mol=target_mol, fp_info=target_fp_info)
            target_sub, labeled_target_sub, labeled_target_frag, labeled_target = get_sub_mol(mol=target_mol,
                                                                                          atom_idx=updated_target_atom_idx)

            if self.query.HasSubstructMatch(target_sub, useChirality=True):
                extr_res = label_query_mol(query_mol=self.query,
                                                                    labeled_sub_mol=labeled_target_sub,
                                                                    sub_mol=target_sub,
                                                                    sub_src_mol=target_mol,
                                                                    allow_add_isotope=True)
                if extr_res:
                    labeled_query_sub_mol, labeled_query_frag , labeled_query= extr_res
                    return labeled_query_sub_mol, labeled_query_frag,labeled_query, labeled_target_sub, labeled_target_frag, labeled_target
                else:
                    return None
            else:
                return None

    