import random
import re
from typing import Tuple

from rdkit import Chem


def smi_tokenizer(smi: str) -> str:
    """Tokenize SMILES

    Args:
        smi (str): SMILES

    Returns:
        str: tokenized SMILES
    """

    smi_pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(smi_pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def canonicalize_smiles(smiles: str, isomeric=True) -> str:
    """Get SMILES strings in canonical form

    Args:
        smiles (str): a SMILES string of a molecule or reaction
        isomeric (bool, optional): with stereo information. Defaults to True.

    Returns:
        str: a canonical SMILES string of a molecule or reaction
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric)
    else:
        return None


def get_isotopic(smi):
    pattern = re.compile(r'\[[0-9][A-Za-z0-9]*\]')
    return pattern.findall(smi)


def convert(src_sub: str, src_frag: str, tgt_frag: str) -> Tuple[str]:
    """Convert SMILES strings of substrutures and fragments
    into the tokenized src(source) input and tgt(target) output of Transformer models.

    Args:
        src_sub (str): substructure SMILES of src        
        src_frag (str): fragment SMILES of src
        tgt_frag (str): fragment SMILES of tgt

    Returns:
        Tuple[str]: a tuple of tokenized src and tgt SMILES strings
    """
    if src_sub:
        sub_smi = smi_tokenizer(src_sub)
        tgt_frag_smi = smi_tokenizer(tgt_frag)

        tgt_smi = ' | ' + tgt_frag_smi
        src_smi = sub_smi + ' | ' + smi_tokenizer(src_frag)
       

        return src_smi, tgt_smi

    else:
        src_smi = ' | ' + smi_tokenizer(src_frag)
        tgt_smi = ' | ' + smi_tokenizer(tgt_frag)

        return src_smi, tgt_smi


def postprocess(smi):
    """Remove extra '[H]' from SMILES of merged mols.

    Args:
        smi (string): SMILES string

    Return:
        post_smi (string): post-processed SMILES string
    """
    smi_pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(smi_pattern)
    tokens = [t for t in regex.findall(smi)]

    post_tokens = []

    for t in tokens:
        # Remove extra '[H]' from 'C' and 'c'
        if t in {'[CH]', '[CH2]', '[CH3]', '[CH4]'}:
            post_tokens.append('C')
        elif t in {'[cH]', '[cH2]', '[cH3]', '[cH4]', '[c]'}:
            post_tokens.append('c')

        # Remove extra '[H]' from 'N', 'n' and 'P'
        elif t in {'[NH]', '[NH2]', '[NH3]', '[N]'}:
            post_tokens.append('N')
        elif t in {'[nH2]', '[nH3]', '[n]'}:
            post_tokens.append('n')
        elif t in {'[PH]', '[P]', '[PH2]', '[PH3]'}:
            post_tokens.append('P')

        # Remove extra '[H]' from 'O', 'o' and 'S'
        elif t in {'[OH]', '[OH2]'}:
            post_tokens.append('O')
        elif t in {'[o]'}:
            post_tokens.append('o')
        elif t in {'[S]', '[SH]', '[SH2]'}:
            post_tokens.append('S')

        # Remove extra '[H]' from 'B'
        elif t in {'[BH]', '[BH2]' ,'[BH3]'}:
            post_tokens.append('B')

        # Remove extra '[H]' from 'F', 'Cl', 'Br', 'I'
        elif t in {'[FH]'}:
            post_tokens.append('F')
        elif t in {'[ClH]'}:
            post_tokens.append('Cl')
        elif t in {'[BrH]'}:
            post_tokens.append('Br')
        elif t in {'[IH]'}:
            post_tokens.append('I')

        else:
            post_tokens.append(t)

    post_smi = ''.join(post_tokens)

    # Replace some specific mols
    # post_smi = post_smi.replace('II', '[131I][131I]')
    post_smi = post_smi.replace('CC1(C)CCCC(C)(C)N1O', 'CC1(C)CCCC(C)(C)N1[O]')
    post_smi = post_smi.replace('Bc1cccc(N)c1', '[B]c1cccc(N)c1')
    post_smi = post_smi.replace('COP=O', 'CO[PH2]=O')

    return post_smi


def get_random_smiles(input_smi):
    if not input_smi:
        return input_smi
    split_input_mol = [Chem.MolFromSmiles(smiles) for smiles in input_smi.split(".")]
    random.shuffle(split_input_mol)
    new_smi = ".".join([Chem.MolToSmiles(mol, doRandom = True) for mol in split_input_mol])
    assert len(new_smi)>0, 'new SMILES is empty string'
    return new_smi    
        