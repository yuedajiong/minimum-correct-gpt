import torch
class TpjWorld(torch.utils.data.Dataset):
    def __init__(self, split, ndigit=(2,3)[0]):
        self.ndigit = ndigit
        number_total, number_valid = (10**self.ndigit)**2, int((10**self.ndigit)**2*0.2)
        permutation = torch.randperm(number_total, generator=torch.Generator())  #generator.manual_seed(122333)  #fixed
        self.indexes = permutation[:number_valid] if split=='valid' else permutation[number_valid:]

    def get_vocab_size(self):
        return 10  #digits: {0..9}

    def get_block_size(self):
        return 3*self.ndigit+1-1  #a,b,a+b, and +1 due to potential carry overflow, but then also -1 because very last digit doesn't ever plug back as there is no explicit <EOS> token to predict, it is implied

    def __len__(self):
        return self.indexes.nelement()

    def __getitem__(self, index):
        index = self.indexes[index].item()
        nd = 10**self.ndigit
        a, b = index//nd, index%nd
        c = a + b
        astr = (f'%0{self.ndigit}d' % a)
        bstr = (f'%0{self.ndigit}d' % b)
        cstr = (f'%0{self.ndigit+1}d' % c)[::-1]  #reverse c to make addition easier, +1 means carry-overflow
        encoded = [int(s) for s in (astr + bstr + cstr)]  #convert each character to its token index
        x = torch.tensor(encoded[:-1], dtype=torch.long)  #x is input to GPT
        #x[self.ndigit*2:] = 10  #if don't comment this，means 2 2-digit adding-number, and part of sum, so learn faster; if not comment, get_vocab_size：10+1
        y = torch.tensor(encoded[1:], dtype=torch.long)   #y is the associated expected outputs, predict the next token in the sequence
        y[:self.ndigit*2-1] = -1  #only train in the output locations. -1 will mask loss to zero：cross_entropy(..., ignore_index=-1)
        return x, y

class TpjBrain(torch.nn.Module):
    class Block(torch.nn.Module):
        class CausalSelfAttention(torch.nn.Module):  #multi-head masked self-attention -> projection
            def __init__(self, n_embd, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1):
                super().__init__()
                self.n_embd = n_embd
                self.n_head = n_head
                self.c_attn = torch.nn.Linear(self.n_embd, self.n_embd * 3)
                self.attn_dropout = torch.nn.Dropout(attn_pdrop)
                self.c_proj = torch.nn.Linear(self.n_embd, self.n_embd)
                self.resid_dropout = torch.nn.Dropout(resid_pdrop)
                self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))  #causal mask to ensure that attention is only applied to the left in the input sequence

            def forward(self, x):
                B, T, C = x.size()  #batch-size, sequence-length, embedding-dimensionality (n_embd)
                q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
                k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  #(B, nh, T, hs)
                q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  #(B, nh, T, hs)
                v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  #(B, nh, T, hs)
                att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1))**0.5)  #causal self-attention; self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = torch.torch.nn.functional.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v  #(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                y = y.transpose(1, 2).contiguous().view(B, T, C)  #re-assemble all head outputs side by side
                y = self.resid_dropout(self.c_proj(y))  #output projection
                return y

        class GELU(torch.nn.Module):  #Gaussian Error Linear Units (GELU) https://arxiv.org/abs/1606.08415
            def forward(self, x):
                return 0.5 * x * (1.0 + torch.tanh((2.0/torch.pi)**0.5 * (x + 0.044715 * torch.pow(x, 3.0))))

        def __init__(self, n_embd, n_head, block_size, resid_pdrop=0.1):
            super().__init__()
            self.ln_1 = torch.nn.LayerNorm(n_embd)
            self.attn = self.__class__.CausalSelfAttention(n_embd, n_head, block_size, resid_pdrop=resid_pdrop)
            self.ln_2 = torch.nn.LayerNorm(n_embd)
            self.mlp = torch.nn.ModuleDict(dict(c_fc=torch.nn.Linear(n_embd, 4 * n_embd), c_proj=torch.nn.Linear(4 * n_embd, n_embd), act=self.__class__.GELU(), dropout=torch.nn.Dropout(resid_pdrop)))
            self.mlpf = lambda x: self.mlp.dropout(self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(x))))

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlpf(self.ln_2(x))
            return x

    def __init__(self, vocab_size, block_size, model_type='gpt-nano', n_embd=48, embd_pdrop=0.1, n_layer=3, n_head=3, ):
        super().__init__()
        self.block_size = block_size
        self.transformer = torch.nn.ModuleDict(dict(
            wte=torch.nn.Embedding(vocab_size, n_embd),
            wpe=torch.nn.Embedding(block_size, n_embd),
            drop=torch.nn.Dropout(embd_pdrop),
            h=torch.nn.ModuleList([self.__class__.Block(n_embd, n_head, block_size) for _ in range(n_layer)]),
            ln_f=torch.nn.LayerNorm(n_embd)))
        self.lm_head = torch.nn.Linear(n_embd, vocab_size, bias=False)
        def init_weights(module):
            if isinstance(module, torch.nn.Linear):
                torch.torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, torch.nn.LayerNorm):
                torch.torch.nn.init.zeros_(module.bias)
                torch.torch.nn.init.ones_(module.weight)
        self.apply(init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):  #a special scaled init to the residual projections
                torch.torch.nn.init.normal_(p, mean=0.0, std=0.02/((2 * n_layer)**0.5))

    def forward(self, idx, targets=None):
        pos = torch.arange(0, idx.size()[1], dtype=torch.long, device=idx.device).unsqueeze(0)  #shape (1, t)
        tok_emb = self.transformer.wte(idx)  #token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  #position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def learn(self, idx, targets):
        logits = self.forward(idx)
        loss = torch.torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def infer(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):  #take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete the sequence max_new_tokens times, feeding the predictions back into the model each time.
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]  #if the sequence context is growing too long we must crop it at block_size
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature  #pluck the logits at the final step and scale by desired temperature
            if top_k is not None:  #optionally crop the logits to only the top k options
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.torch.nn.functional.softmax(logits, dim=-1)  #apply softmax to convert logits to (normalized) probabilities
            if do_sample:  #either sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            else:  #or take the most likely element
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)  #append sampled index to the running sequence and continue
        return idx

class TpjLearn:
    def __init__(self, brain, train_dataset, valid_dataset, weight_decay=0.1, betas=(0.9, 0.95)):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.brain = brain.to(self.device)
        self.train_dataset, self.valid_dataset = train_dataset, valid_dataset
        no_decay, do_decay = set(), set()
        for mn, m in brain.named_modules():
            for pn, p in m.named_parameters():   #named_modules is recursive, use set() to filter
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias') or (pn.endswith('weight') and isinstance(m, (torch.torch.nn.LayerNorm, torch.torch.nn.Embedding))):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, (torch.torch.nn.Linear)):
                    do_decay.add(fpn)
        param_dict = {pn: p for pn, p in brain.named_parameters()}
        optim_groups = [{"params": [param_dict[pn] for pn in sorted(list(do_decay))], "weight_decay": weight_decay}, {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}]
        self.optimizer = torch.optim.AdamW(optim_groups, lr=5e-4, betas=betas)  #self.optimizer = torch.optim.AdamW(brain.parameters(), lr=5e-4, betas=betas)

    def valid(self, dataset):
        ndigit = self.train_dataset.ndigit
        results = []
        factors = torch.tensor([[10**i for i in range(ndigit+1)][::-1]]).to(self.device)
        loader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=64, num_workers=1, drop_last=False)
        for b, (x, y) in enumerate(loader):
            x = x.to(self.device)
            d1d2 = x[:, :ndigit*2]  #isolate the first two digits of the input sequence alone
            d1d2d3 = self.brain.infer(d1d2, ndigit+1, do_sample=False) # using greedy argmax, not sampling
            d3 = d1d2d3[:, -(ndigit+1):]  #isolate the last digit of the sampled sequence
            d3 = d3.flip(1) #reverse the digits to their "normal" order
            d1i = (d1d2[:,:ndigit] * factors[:,1:]).sum(1)   # decode the integers from individual digits
            d2i = (d1d2[:,ndigit:ndigit*2] * factors[:,1:]).sum(1)  # decode the integers from individual digits
            d3i_pred = (d3 * factors).sum(1)
            d3i_gt = d1i + d2i #manually calculate the ground truth
            correct = (d3i_pred == d3i_gt).cpu()
            for i in range(x.size(0)):
                results.append(int(correct[i]))
        rt = torch.tensor(results, dtype=torch.float)
        print("valid correct: %d/%d = %.2f%%"%(rt.sum(), len(results), 100*rt.mean()))
        return rt.mean()

    def learn(self, grad_norm_clip=1.0, max_iters=5000):
        train_loader = torch.utils.data.dataloader.DataLoader(self.train_dataset, batch_size=64, num_workers=1, drop_last=False, pin_memory=True, shuffle=False, sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)))
        top_score = 0.0
        data_iter = iter(train_loader)
        for iter_num in range(max_iters):
            batch = [t.to(self.device) for t in next(data_iter)]
            x, y = batch
            logits, self.loss = self.brain.learn(x, y)
            self.brain.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.torch.nn.utils.clip_grad_norm_(self.brain.parameters(), grad_norm_clip)
            self.optimizer.step()
            if iter_num % 500 == 0:
                train_max_batches = {1: None, 2: None, 3: 5}[self.train_dataset.ndigit] # if ndigit=2 we can afford the whole train set, ow no
                self.brain.eval()
                with torch.no_grad():
                    train_score = self.valid(self.train_dataset)
                    valid_score  = self.valid(self.valid_dataset)
                score = train_score + valid_score
                if score > top_score:
                    top_score = score
                    print("iter", iter_num, "save model, top score %.4f"%(score))
                    import os
                    ckpt_path = os.path.join('./chkp/', "model.pt")
                    if not os.path.exists(ckpt_path): os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    torch.save(self.brain.state_dict(), ckpt_path)
                self.brain.train()
                if score==1.0*2: break

if __name__ == '__main__':
    train_dataset = TpjWorld(split='train')
    valid_dataset = TpjWorld(split='valid')
    brain = TpjBrain(train_dataset.get_vocab_size(), train_dataset.get_block_size())
    learn = TpjLearn(brain, train_dataset, valid_dataset)
    learn.learn()
