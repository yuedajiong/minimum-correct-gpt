import torch

class Add(torch.utils.data.Dataset):
    def __init__(self, split, ndigit=(2,3)[0]):
        self.ndigit = ndigit
        number_total, number_valid = (10**self.ndigit)**2, int((10**self.ndigit)**2 * 0.2)
        permutation = torch.randperm(number_total, generator=torch.Generator())  #generator.manual_seed(122333)  #fixed
        self.indexes = permutation[:number_valid] if split=='valid' else permutation[number_valid:]

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

    def get_vocab_size(self):
        return 10  #digits: {0..9}

    def get_block_size(self):
        return 3*self.ndigit+1-1  #a,b,a+b, and +1 due to potential carry overflow, but then also -1 because very last digit doesn't ever plug back as there is no explicit <EOS> token to predict, it is implied

class Atte(torch.nn.Module):
    def __init__(self, d_model, nhead, d_feedforward, batch_first, dropout=0.1, norm_first=True, layer_norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.norm_first = norm_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.mha = torch.nn.modules.activation.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.dropout1 = torch.nn.Dropout(dropout)
        #
        if d_feedforward:
            self.ffw = torch.nn.Sequential(torch.nn.Linear(d_model, d_feedforward), torch.nn.ReLU(), torch.nn.Linear(d_feedforward, d_model))
            self.dropout2 = torch.nn.Dropout(dropout)
            self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        a = x
        if self.norm_first:
           a = self.norm1(a)
        a = self.mha(a, a, a, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        a = self.dropout1(a)
        o = x + a
        if not self.norm_first:
            o = self.norm1(o)
        #
        if self.ffw is not None:
            f = self.ffw(o)
            f = self.dropout2(f)
            o = o + f
            o = self.norm2(o)
        return o

class Embd(torch.nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab_size, n_embd).to(device)
        self.wpe = torch.nn.Embedding(block_size, n_embd).to(device)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, idx):
        if 1:
            pos = torch.arange(0, idx.size()[1], dtype=torch.long, device=idx.device).unsqueeze(0)  #[0,1,2,3,4,5,]
            tok_emb = self.wte(idx)
            pos_emb = self.wpe(pos)
            o = self.drop(tok_emb + pos_emb)
            return o
        elif 0:
            class PositionalEncoding(nn.Module):
                def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
                    super().__init__()
                    self.dropout = nn.Dropout(p=dropout)
                    position = torch.arange(max_len).unsqueeze(1)
                    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
                    pe = torch.zeros(max_len, 1, d_model)
                    pe[:, 0, 0::2] = torch.sin(position * div_term)
                    pe[:, 0, 1::2] = torch.cos(position * div_term)
                    self.register_buffer('pe', pe)

                def forward(self, x: Tensor) -> Tensor:  #x: [seq_len, batch_size, embedding_dim]
                    x = x + self.pe[:x.size(0)]
                    return self.dropout(x)

            pos_encoder = PositionalEncoding(d_model, dropout)
            src = self.pos_encoder(src)
            return dat
        else:
            def position_encoding(seq_len, dim_model, device):
                pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
                dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
                phase = pos / (1e4 ** torch.div(dim, dim_model, rounding_mode="floor"))
                return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))
            tok_emb = self.wte(idx)
            pos_emb = self.wpe(pos)
            seq_len, dimension = dat.size(1), dat.size(2)
            dat += position_encoding(seq_len, dimension, device)
            return dat

class Task(torch.nn.Module):
    def __init__(self, n_embd, vocab_size, device=None, dtype=None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(n_embd)
        self.head = torch.nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, out, decode):
        out = self.norm(out)
        logits = self.head(out)
        if not decode:
            return logits
        else:
            probs = torch.torch.nn.functional.softmax(logits, dim=-1)
            _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx_next = torch.squeeze(idx_next, dim=-1)
            return logits, idx_next

class Mind(torch.nn.Module):
    def __init__(self, vocab_size, block_size, batch_first=True, hidden_dimension=512, nhead=16, device=None):
        super().__init__()
        self.embd = Embd(vocab_size=vocab_size, block_size=block_size, n_embd=hidden_dimension).to(device)
        self.core = Core(d_model=hidden_dimension, nhead=nhead, batch_first=batch_first).to(device)
        self.task = Task(n_embd=hidden_dimension, vocab_size=vocab_size).to(device)

    def forward(self, I, decode):
        E = self.embd(I)
        C = self.core(E)
        T = self.task(C, decode=decode)
        return T

def main(pretrain_file=None, checkpoint_file='./temp/ckpt/checkpoint.pth', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    train_dataset_txt = Add('train')
    valid_dataset_txt = Add('valid')
 
    network = Mind(vocab_size=train_dataset_txt.get_vocab_size(), block_size=train_dataset_txt.get_block_size(), )

    if pretrain_file is not None and os.path.exists(pretrain_file):
        state_dict = torch.load(pretrain_file)
        print('load pretrain', pretrain_file)
    network = network.to(device)

    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, betas=(0.9,0.999), weight_decay=0.01, eps=1e-08, )

    train_dataloader_txt = torch.utils.data.dataloader.DataLoader(train_dataset_txt, batch_size=32, num_workers=1, drop_last=False, pin_memory=False, collate_fn=None, shuffle=1, sampler=None) 
    valid_dataloader_txt = torch.utils.data.dataloader.DataLoader(valid_dataset_txt, batch_size=32, num_workers=1, drop_last=False, pin_memory=False, collate_fn=None, shuffle=0, sampler=None)   
    best_train_loss = None
    save_train_step = None
    epochs = 10
    for epoch in range(epochs):
        for index, (X,Y) in enumerate(train_dataloader_txt):
            I = X.to(device)
            T = Y.to(device)
  
            logits = network(I, decode=False)

            loss_txt_predict_next_cross_entropy_logits = torch.torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), T.view(-1), ignore_index=-1)
            loss = loss_txt_predict_next_cross_entropy_logits

            network.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)

            if best_train_loss is None or loss.item() < best_train_loss:
                if 0:
                    if not os.path.exists(checkpoint_file): os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
                    torch.save(network, checkpoint_file)
                best_train_loss = loss.item()
                #print("epoch=%06d  index=%06d  loss=%.4f  best_train_loss=%.4f  save checkpoint"%(epoch, index, loss.item(), (-1 if best_train_loss is None else best_train_loss)))

            if 0:
                if epoch==0 and index==0:
                    import torchviz  #pip install torchviz
                    torchviz.make_dot(loss).render(filename="network", directory="./temp/", format="svg", view=False, cleanup=True, quiet=True)

            if (epoch*len(train_dataloader_txt)+index)%100==0: 
                print('epoch=%06d  index=%06d  loss=%.6f  best_train_loss=%.6f  >>>>'%(epoch, index, loss.item(), (-1 if best_train_loss is None else best_train_loss)))

                if 1:
                    with torch.no_grad():
                        network.train()
                        losses = []
                        for index, (X,Y) in enumerate(valid_dataloader_txt):
                            I = X.to(device)
                            T = Y.to(device)
                            logits, D = network(I, decode=True)
                            loss = torch.torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), T.view(-1), ignore_index=-1)
                            losses.append(loss.item())
                            if 0:
                                x0 = I[0].detach().cpu().tolist()
                                y0 = T[0].detach().cpu().tolist()
                                d0 = D[0].detach().cpu().tolist()
                                xx = ''.join([str(int(i)) for i in x0])
                                yy = ''.join([str(int(i)) for i in y0])
                                dd = ''.join([str(int(i)) for i in d0])
                                print(xx+' -> '+' '+yy+' ?= '+dd+'\n')
                        print('epoch=%06d  losses=%.6f valid'%(epoch, sum(losses)/len(losses)))
                        network.eval()

if __name__ == '__main__':
    import signal,os; signal.signal(signal.SIGINT, lambda self,code: os._exit(0))
    main()
