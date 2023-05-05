import torch

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, n_embd, n_head, block_size, use_bias):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.c_attn = torch.nn.Linear(n_embd, 3 * n_embd, bias=use_bias)
        self.c_proj = torch.nn.Linear(n_embd, n_embd, bias=use_bias)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        batch, sequence, embed = x.size()
        q, k ,v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(batch, sequence, self.n_head, embed // self.n_head).transpose(1, 2)
        k = k.view(batch, sequence, self.n_head, embed // self.n_head).transpose(1, 2)
        v = v.view(batch, sequence, self.n_head, embed // self.n_head).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1))))
        a = a.masked_fill(self.bias[:,:,:sequence,:sequence] == 0, float('-inf'))
        a = torch.nn.functional.softmax(a, dim=-1)
        y = a @ v
        y = y.transpose(1, 2).contiguous().view(batch, sequence, embed)
        y = self.c_proj(y)
        return y

class ResidualBlock(torch.nn.Module):
    def __init__(self, n_embd, n_head, block_size, use_bias):
        super().__init__()
        self.lna = torch.nn.LayerNorm(n_embd)
        self.att = CausalSelfAttention(n_embd, n_head, block_size, use_bias)
        self.lnb = torch.nn.LayerNorm(n_embd)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(n_embd, 4 * n_embd, bias=use_bias), torch.nn.GELU(), torch.nn.Linear(4 * n_embd, n_embd, bias=use_bias))

    def forward(self, x):
        x = x + self.att(self.lna(x))
        x = x + self.mlp(self.lnb(x))
        return x

class GPT(torch.nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, use_bias):
        super().__init__()
        self.te = torch.nn.Embedding(vocab_size, n_embd)
        self.pe = torch.nn.Embedding(block_size, n_embd)
        self.hi = torch.nn.Sequential(*[ResidualBlock(n_embd, n_head, block_size, use_bias) for _ in range(n_layer)])
        self.ln = torch.nn.LayerNorm(n_embd)
        self.lm = torch.nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        tok_emb = self.te(idx)
        pos_emb = self.pe(torch.arange(0, idx.size()[1], dtype=torch.long, device=idx.device).unsqueeze(0))
        x = tok_emb + pos_emb
        x = self.hi(x)
        logits = self.lm(self.ln(x)[:, -1, :]) 
        return logits

def main(seq=list(map(int, "11100110110")), vocab_size=2, block_size=3, n_embd=16, n_head=4, n_layer=4, ):
    x, y = [], []
    for i in range(len(seq)-block_size):
        x.append(seq[i:i+block_size])
        y.append(seq[i+block_size])
        print('sample:', x[-1], '-->', y[-1])
    X, Y = torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    network = GPT(vocab_size, block_size, n_embd, n_head, n_layer, use_bias=False)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.1)
    for epoch in range(1, 100+1):
        logits = network(X)
        loss = torch.nn.functional.cross_entropy(logits, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch%10==0: print('train:', 'epoch=%03d'%(epoch), 'loss=%.4f'%(loss.item()))

    def all_possible_full_permutation(n, k):
        if k == 0:
            yield []
        else:
            for i in range(n): 
                for c in all_possible_full_permutation(n, k-1): yield [i] + c
    for x in all_possible_full_permutation(vocab_size, block_size):       
        logits = network(torch.tensor(x, dtype=torch.long)[None, ...])
        probs = torch.nn.functional.softmax(logits, dim=-1)
        print('infer:', x, '-->', torch.argmax(probs[0]).tolist(), '    ', [round(p,2) for p in probs[0].tolist()])

if __name__ == '__main__':
    torch.manual_seed(333)
    main()
