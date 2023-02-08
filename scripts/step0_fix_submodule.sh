#!/bin/bash

# fix typo
sed -i 's/type=str, default=250/type=int, default=250/g'  RetrievalModel/prepare.py 

# move tensor to cpu in case of cuda OOM
sed -i 's/q = torch.cat(q_list, dim=0)/q = torch.cat(q_list, dim=0).to("cpu")/g'  RetrievalModel/pretrain.py 
sed -i 's/r = torch.cat(r_list, dim=0)/r = torch.cat(r_list, dim=0).to("cpu")/g'  RetrievalModel/pretrain.py 
sed -i 's/gold = torch.arange(bsz, device=scores.device)/gold = torch.arange(bsz, device="cpu")/g' RetrievalModel/pretrain.py 

# support more recent pytorch version
sed -i 's/q \*= self.scaling/q = q.contiguous() \* self.scaling/g'  RetrievalModel/transformer.py


# We use token type for valid data, otherwise we might have OOM with large batch size. Alternatively, we could set valid_batch_size to a small number(which might be less efficient).
sed -i 's/batch_size = opt.batch_size if is_train else opt.valid_batch_size/batch_size = opt.batch_size/g'  MolecularTransformer/onmt/inputters/inputter.py
sed -i 's/if is_train and opt.batch_type == "tokens"/if opt.batch_type == "tokens"/g'  MolecularTransformer/onmt/inputters/inputter.py