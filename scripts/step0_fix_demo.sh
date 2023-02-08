#!/bin/bash
# make the demo runnable on CPU 
sed -i 's/device =/# device =/g'  RetrievalModel/search_index.py
sed -i 's/model.to(device)/# model.to(device)/g'  RetrievalModel/search_index.py
sed -i 's/mips.to_gpu()/# mips.to_gpu()/g'  RetrievalModel/search_index.py
sed -i 's/model.cuda()/# model.cuda()/g'  RetrievalModel/search_index.py
sed -i "s/q = move_to_device(batch, torch.device('cuda')).t()/q = torch.from_numpy(batch).contiguous().t() /g"  RetrievalModel/search_index.py
sed -i 's/model = torch.nn.Data/# model = torch.nn.Data/g'  RetrievalModel/search_index.py