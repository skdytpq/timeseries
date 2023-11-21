"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
from data_ett import *
import pdb 
from argsep import get_args
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

 
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)
args = get_args()
def _get_data(flag):
    data_dict = {
        'ETTh1':Dataset_ETT_hour,
        'ETTh2':Dataset_ETT_hour,
        'ETTm1':Dataset_ETT_minute,
        'ETTm2':Dataset_ETT_minute,
        'WTH':Dataset_Custom,
        'ECL':Dataset_Custom,
        'Solar':Dataset_Custom,
        'custom':Dataset_Custom,
    }
    Data = data_dict[args.data]
    timeenc = 0 if args.embed!='timeF' else 1

    if flag == 'test':
        shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
    elif flag=='pred':
        shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq # 
    data_set = Data(
        root_path=args.root_path, #
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len], # 24*4*4 24*4 24*4
        features=args.features, # S
        target=args.target, # OT
        inverse=args.inverse, # False
        timeenc=timeenc, 
        freq=freq, # 0 train, 
        cols=args.cols
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader

train_data, train_loader = _get_data(flag = 'train')
vali_data, vali_loader = _get_data(flag = 'test')
test_data, test_loader = _get_data(flag = 'pred')
x_size = 24*4*2 # dataset마다 다르게 설정하기
y_size = 24*4*2
max_len = 128
model = Transformer(
                    d_model=d_model,
                    x_size=x_size,
                    y_size=y_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
criterion_nll = nn.NLLLoss()
window_size = 192
stride = 50

def train(model, traindata, optimizer):
    model.train()
    epoch_loss = 0
    for i,(seq_x,seq_y,_,__)in enumerate(traindata): # 
        loss = 0.0
        
        x = seq_x # batch X 24*4*2
        y = seq_y # batch X 24*4*2
        train_batch = x[0]
        optimizer.zero_grad()
        # Window 구성하기 # model에 hidden 과 cell이 들어가야 함.
        attn, p_attn ,x_hat, gen_mean, gen_var, y_hat, inf_mean, inf_var = model(x, y) # 해당 모델에 다 넣기 
        kl = torch.mean(torch.log(gen_var/inf_var) + (inf_var + torch.square(inf_mean - gen_mean))/(2*gen_var))  
        recon_enc = torch.mean(torch.square(x_hat[:, -1, :, :] - x))
        recon_dec = torch.mean(torch.square(y_hat[:, -1, :, :] - y))
        loss += kl +recon_enc + recon_dec
        loss += criterion_nll(p_attn,attn) 
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(train_batch)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(train_batch)
train(model, train_loader  , optimizer )

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            # 
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    dataset , dataloader = _get_data('train')
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


#if __name__ == '__main__':
#    run(total_epoch=epoch, best_loss=inf)
