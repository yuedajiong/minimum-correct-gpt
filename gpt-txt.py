import torch
import os

import torchvision  #pip install torchvision
import PIL
class TpjImg(torch.utils.data.Dataset):
    def __init__(self, input_path=['./data/img/'][0]):
        def list_image(path_name):
            images = sorted([os.path.join(path_name, file_name) for file_name in os.listdir(path_name) if os.path.isfile(os.path.join(path_name, file_name))])
            return images[:]

        def load_image(file_name, transform):
            image = PIL.Image.open(file_name).convert('RGB')
            tensor = transform(image)
            return tensor

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.buffer = [load_image(file_name, transform) for file_name in list_image(input_path)]

    def __getitem__(self, index):
        return self.buffer[index]

    def __len__(self):
        return len(self.buffer)

import numpy
class TpjTxt(torch.utils.data.Dataset):
    def __init__(self, is_valid_index, block_size, input_file=['./data/txt/Dream_of_the_Red_Mansion.txt'][0], train_valid_files=['./temp/buff/data_train.bin','./temp/buff/data_valid.bin'], vocab_file='./temp/buff/data_vocab.dat', use_tiktoken=False):
        cache_file = train_valid_files[is_valid_index]
        if not os.path.exists(cache_file):
            def make_data(i_file, o_file, train_valid_files, is_valid_index):
                with open(input_file, 'r') as obj:
                    data = obj.read()
                train_valid_data = (data[:int(len(data)*1.0)], data[int(len(data)*0.0):])  #John full-data-as-trainset-and-validset
                os.makedirs(os.path.dirname(o_file), exist_ok=True)               
                if use_tiktoken:
                    import tiktoken  #pip install tiktoken
                    encoder = tiktoken.get_encoding("gpt2")  #BPE
                    numpy.array(encoder.encode_ordinary(train_valid_data[is_valid_index]), dtype=numpy.uint16).tofile(train_valid_files[is_valid_index])
                    numpy.array([50304], dtype=numpy.uint16).tofile(vocab_file)  #vocab_size=50257@gpt
                else:
                    self.vocab_data = sorted(list(set(data)))
                    self.vocab_size = len(self.vocab_data)
                    self.stoi = { c:i for i,c in enumerate(self.vocab_data) }
                    self.itos = { i:c for i,c in enumerate(self.vocab_data) }
                    numpy.array([self.stoi[c] for c in train_valid_data[is_valid_index]], dtype=numpy.uint16).tofile(train_valid_files[is_valid_index])
                    if not os.path.exists(vocab_file):
                        with open(vocab_file, 'w') as file:
                            file.write(''.join([c for c in self.vocab_data]))
                            #print('vocab size:', self.vocab_size)
            make_data(input_file, cache_file, train_valid_files, is_valid_index)
        else:
            with open(vocab_file, 'r') as file:
                self.vocab_data = file.read()
            self.vocab_size = len(self.vocab_data)
            self.stoi = { ch:i for i,ch in enumerate(self.vocab_data) }
            self.itos = { i:ch for i,ch in enumerate(self.vocab_data) }
        self.cache_data = numpy.memmap(cache_file, mode='r', dtype=numpy.uint16)
        self.block_size = block_size

    def __len__(self):
        return len(self.cache_data) - self.block_size

    def __getitem__(self, index):
        x = torch.from_numpy((self.cache_data[index+0:index+0+self.block_size]).astype(numpy.int64))
        y = torch.from_numpy((self.cache_data[index+1:index+1+self.block_size]).astype(numpy.int64))
        return x, y

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

class TpjBrain(torch.nn.Module):
    class AttentionMLP(torch.nn.Module):
        class CausalSelfAttention(torch.nn.Module):  #multi-head-masked-self-attention -> projection
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
                q, k ,v = self.c_attn(x).split(self.n_embd, dim=2)
                q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  #(B, nh, T, hs)
                k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  #(B, nh, T, hs)
                v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  #(B, nh, T, hs)
                att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1))**0.5)  #causal-self-attention; self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = torch.torch.nn.functional.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v  #(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                y = y.transpose(1, 2).contiguous().view(B, T, C)  #re-assemble all head outputs side by side
                y = self.resid_dropout(self.c_proj(y))  #output projection
                return y

        class GELU(torch.nn.Module):  #GaussianErrorLinearUnits  https://arxiv.org/abs/1606.08415
            def forward(self, x):
                return 0.5 * x * (1.0 + torch.tanh((2.0/torch.pi)**0.5 * (x + 0.044715 * torch.pow(x, 3.0))))

        def __init__(self, n_embd, n_head, block_size, resid_pdrop=0.1):
            super().__init__()
            self.att = torch.nn.Sequential(torch.nn.LayerNorm(n_embd), self.__class__.CausalSelfAttention(n_embd, n_head, block_size, resid_pdrop=resid_pdrop))
            self.mlp = torch.nn.Sequential(torch.nn.LayerNorm(n_embd), torch.nn.Linear(n_embd, 4 * n_embd), self.__class__.GELU(), torch.nn.Linear(4 * n_embd, n_embd), torch.nn.Dropout(resid_pdrop))

        def forward(self, x):
            x = x + self.att(x)
            x = x + self.mlp(x)
            return x

    def __init__(self, vocab_size, block_size, n_embd, embd_pdrop, n_layer, n_head, ):
        super().__init__()
        self.wte=torch.nn.Embedding(vocab_size, n_embd)
        self.wpe=torch.nn.Embedding(block_size, n_embd)
        self.drop=torch.nn.Dropout(embd_pdrop)
        self.blocks = torch.nn.ModuleList([self.__class__.AttentionMLP(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.norm = torch.nn.LayerNorm(n_embd)
        self.head = torch.nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        pos = torch.arange(0, idx.size()[1], dtype=torch.long, device=idx.device).unsqueeze(0)  #shape (1, t)
        tok_emb = self.wte(idx)  #token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  #position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits

def main(pretrain_file=None, checkpoint_file='./temp/ckpt/checkpoint.pth', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    if 0:
        config_img = dict(img_size=[224], in_chans=3, num_output=0, patch_size=8, embed_dim=768, num_heads=12)
        train_dataset_img = TpjImg()

    config_txt = dict(block_size=1024//2)
    train_dataset_txt = TpjTxt(is_valid_index=0, block_size=config_txt['block_size'])
    valid_dataset_txt = TpjTxt(is_valid_index=1, block_size=config_txt['block_size'])
 
    network = TpjBrain(train_dataset_txt.get_vocab_size(), train_dataset_txt.get_block_size(), n_embd=768//2, embd_pdrop=0.0, n_layer=8, n_head=6).to(device)
    if pretrain_file is not None and os.path.exists(pretrain_file):
        state_dict = torch.load(pretrain_file)
        print('use pretrain', pretrain_file)
    network = network.to(device)

    optimizer = torch.optim.AdamW(network.parameters(), lr=0.0005)

    dataloader_txt = torch.utils.data.dataloader.DataLoader(train_dataset_txt, batch_size=16, num_workers=1, drop_last=False, pin_memory=False, collate_fn=None, shuffle=True, sampler=None)  
    best_train_loss = None
    save_train_step = None
    epochs = 1*100
    print('len(dataloader_txt)', len(dataloader_txt))
    for epoch in range(epochs):
        for index, (X,Y) in enumerate(dataloader_txt):
            I = X.to(device)
            T = Y.to(device)
            if 0:
                x0 = I[0].detach().cpu().tolist()
                y0 = T[0].detach().cpu().tolist()
                xx = ''.join([train_dataset_txt.itos[int(i)] for i in x0])
                yy = ''.join([train_dataset_txt.itos[int(i)] for i in y0])
                print(xx+'\n->\n'+' '+yy+'\n')
            O = network(I)

            if 0:
                loss_img_reconstruct_self_mse = torch.nn.functional.mse_loss(O, T)

            loss_txt_predict_next_cross_entropy_logits = torch.torch.nn.functional.cross_entropy(O.view(-1, O.size(-1)), T.view(-1), ignore_index=-1)

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
                print("epoch=%06d  index=%06d  loss=%.4f  best_train_loss=%.4f  save model."%(epoch, index, loss.item(), (-1 if best_train_loss is None else best_train_loss)))
                if 0:
                    network.eval()
                    with torch.no_grad():
                        O = network(I)
                        oo = O.detach().permute(0, 2, 3, 1).cpu().numpy()
                        temp_path = './temp/img/'
                        os.makedirs(temp_path, exist_ok=True)
                        for offset,ooo in enumerate(oo):
                            PIL.Image.fromarray((ooo*255).astype('uint8')).save(temp_path+'/output_{:04d}_{:04d}_{:04d}_0.png'.format(epoch,index,offset))
                    network.train()

            if epoch==0 and index==0:
                import torchviz  #pip install torchviz
                torchviz.make_dot(loss).render(filename="network", directory="./temp/", format="svg", view=False, cleanup=True, quiet=True)
            if (epoch*len(dataloader_txt)+index)%10==0: 
                print('epoch=%06d  index=%06d  loss=%.4f  best_train_loss=%.4f  >>>'%(epoch, index, loss.item(), (-1 if best_train_loss is None else best_train_loss)))

if __name__ == '__main__':
    import signal; signal.signal(signal.SIGINT, lambda self,code: os._exit(0))
    main()
