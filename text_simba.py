# pytorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
# model
from simba import Mmamba, Text_Mmamba
from transformers import AutoTokenizer, T5Tokenizer
from transformers import T5EncoderModel
from torch.optim import AdamW
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
# import dadam as optim
# import cosine_lr_scheduler as lr_scheduler

# others
from glob import glob
import numpy as np
import os
import json
from tqdm import tqdm
import copy
import math
import argparse
import logging
import time
import spacy
# import bcolors

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# torch.set_printoptions(threshold=np.inf)

def parse_opt():
    parser = argparse.ArgumentParser()
    # continue or not
    parser.add_argument("-c", "--is_continue", action="store_true")
    
    # general
    parser.add_argument('--device', type=str,
                        help='gpu device.', default='cuda')
    parser.add_argument('--project_name', type=str,
                        help='project_name.', default='new_project')    
    
    # about model
    parser.add_argument('--layer_num', type=int,
                        help='layers of model', default=24)
    parser.add_argument('--d_state', type=int,
                        help='state size of mamba', default=512)
    parser.add_argument("-i", "--is_inner", action="store_true")
    
    # about training
    parser.add_argument('--batch', type=int,
                        help='batch size', default=4)
    parser.add_argument('--accumulation_step', type=int,
                        help='accumulation_step', default=16)

    # about continue
    parser.add_argument('--ckpt', type=int,
                        help='ckpt epoch', default=None)
    args = parser.parse_args()
    return args

def create_logger(logger_file_path):

    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(logger_file_path, log_name)

    logger = logging.getLogger()  # 设定日志对象
    logger.setLevel(logging.INFO)  # 设定日志等级

    file_handler = logging.FileHandler(final_log_file)  # 文件输出
    console_handler = logging.StreamHandler()  # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)  # 设置文件输出格式
    console_handler.setFormatter(formatter)  # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

opt = parse_opt()
print(opt)

r'''
class WhiteSpaceTokenizer():
    """This tokenizer should be used for natural language descriptions.
    For example:
    ["he didn't, know he's going home.", 'shorter sentence'] =>
    [[78, 62, 31,  4, 78, 25, 19, 34],
    [59, 77,  0,  0,  0,  0,  0,  0]]
    """
    PUNCTUATION = "?:!.,;"

    def __init__(self, n_bins: int, pad_idx: int = 0, language: str = "en_core_web_sm",
                    lemma: bool = True, stopwords: bool = True) -> None:
        self.n_bins = n_bins
        self.pad_idx = pad_idx
        self.lemma = lemma
        self.stopwords = stopwords
        try:
            self.nlp = spacy.load(language)
        except IOError:
            spacy.cli.download(language)  # type: ignore
            self.nlp = spacy.load(language)

    @tp.no_type_check
    def __call__(self, texts: tp.List[tp.Optional[str]],
                    return_text: bool = False) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Take a list of strings and convert them to a tensor of indices.

        Args:
            texts (list[str]): List of strings.
            return_text (bool, optional): Whether to return text as additional tuple item. Defaults to False.
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Indices of words in the LUT.
                - And a mask indicating where the padding tokens are
        """
        output, lengths = [], []
        texts = deepcopy(texts)
        for i, text in enumerate(texts):
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(torch.Tensor([self.pad_idx]))
                lengths.append(0)
                continue

            # convert numbers to words
            text = re.sub(r"(\d+)", lambda x: num2words(int(x.group(0))), text)  # type: ignore
            # normalize text
            text = self.nlp(text)  # type: ignore
            # remove stopwords
            if self.stopwords:
                text = [w for w in text if not w.is_stop]  # type: ignore
            # remove punctuation
            text = [w for w in text if w.text not in self.PUNCTUATION]  # type: ignore
            # lemmatize if needed
            text = [getattr(t, "lemma_" if self.lemma else "text") for t in text]  # type: ignore

            texts[i] = " ".join(text)
            lengths.append(len(text))
            # convert to tensor
            tokens = torch.Tensor([hash_trick(w, self.n_bins) for w in text])
            output.append(tokens)

        mask = length_to_mask(torch.IntTensor(lengths)).int()
        padded_output = pad_sequence(output, padding_value=self.pad_idx).int().t()
        if return_text:
            return padded_output, mask, texts  # type: ignore
        return padded_output, mask
'''

def token_to_seq(tokens):
    '''
    INPUT:
    tokens: a encodec compressed token with 4 residual layers
    1 5 9... (course layer)
    2 6 10...
    3 7 11...
    4 8 12... (fine layer)
    
    OUTPUT:
    a: a flatten seq
    1 2 3 4 5 6 7 8 9 10 11 12...
    '''
    K, L = tokens.shape
    a = np.zeros((K*L))
    for i in range(K*L):
        a[i] = tokens[i%4, i//4]
    return a

def seq_to_token(seq):
    '''
    INPUT:
    a: a flatten seq
    1 2 3 4 5 6 7 8 9 10 11 12...
    
    OUTPUT:
    tokens: a encodec compressed token with 4 residual layers
    1 5 9... (course layer)
    2 6 10...
    3 7 11...
    4 8 12... (fine layer)
    '''
    L = seq.shape[0]
    print(L)
    a = np.zeros((4, L//4))
    idx = 0
    for i in range(L//4):
        for j in range(4):
            a[j][i] = seq[idx]
            idx+=1
    return a

def to_delay(para):
    '''
    INPUT:
    tokens: a k layers RVQ token
    1 5 9... (course layer)
    2 6 10...
    3 7 11...
    4 8 12... (fine layer)
    
    OUTPUT:
    delay: the delay pattern
    0 1 5 9... (course layer)
    0 0 2 6 10...
    0 0 0 3 7 11...
    0 0 0 0 4 8 12... (fine layer)
    '''
    K, L = para.shape
    delay = np.zeros((K, L)) + 2048
    for i in range(K):
        # delay[i, i:] = para[i, :500-i]
        delay[i, i+1:] = para[i, :L-1-i]
    return delay

def to_parallel(delay):
    '''
    INPUT:
    delay: the delay pattern
    1 5 9... (course layer)
    0 2 6 10...
    0 0 3 7 11...
    0 0 0 4 8 12... (fine layer)
    
    
    OUTPUT:
    para: a k layers RVQ token w/ parallel form
    1 5 9... (course layer)
    2 6 10...
    3 7 11...
    4 8 12... (fine layer)
    '''
    K, L = delay.shape
    para = np.zeros((K, L)) + 2048
    for i in range(K):
        # para[i, :500-i] = delay[i, i:]
        para[i, :499-i] = delay[i, i+1:]
    return para

def random_zeroing(batch_tensor, batch_zero_prob=0.1, bit_zero_prob=0.1):
    """
    Args:
    - batch_tensor (torch.Tensor): 输入的0, 1向量，维度为(B, L)
    - batch_zero_prob (float): 每个batch全变成0的概率
    - bit_zero_prob (float): batch中的1变成0的概率
    
    Returns:
    - torch.Tensor: 经过随机置零后的向量
    """
    batch_size, seq_length = batch_tensor.shape

    # Step 1: 每个batch有10%的几率全变成0
    batch_mask = torch.bernoulli(torch.full((batch_size,), 1 - batch_zero_prob))
    # print(batch_mask.bool())
    batch_mask = batch_mask.bool()
    batch_tensor[~batch_mask] = 0
    # print(batch_tensor)

    # Step 2: 对于剩下的batch中的1，以5%的几率变成0
    bit_mask = torch.bernoulli(torch.full_like(batch_tensor.float(), 1 - bit_zero_prob))
    bit_mask = bit_mask.bool()
    # print(bit_mask)
    batch_tensor[batch_mask] = batch_tensor[batch_mask] * bit_mask[batch_mask]

    return batch_tensor

def cal_torch_model_params(model):
    '''
    :param model:
    :return:
    '''
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': total_params/1000000, 'total_trainable_params': total_trainable_params/1000000}

class Dataset(object):
    '''
    Jamendo dataset
    '''
    def __init__(self, datalist) -> None:
        self.data_path = datalist
    
    def __getitem__(self, idx):
        # shape = [1, 128, slice_length]
        path = self.data_path[idx]
        data = np.load(path, allow_pickle=True)
        # return seq[:, 0:1499], seq[:, 1:1500]
        return torch.LongTensor(data[:, :-1]), torch.LongTensor(data[:, 1:])
    
    def __len__(self):
        return len(self.data_path)

class MB_Dataset(object):
    '''
    MusicBench dataset
    '''
    def __init__(self, metadata, root_path, codec_layer = 4, L = 500) -> None:
        self.meta = metadata
        self.root = root_path
        self.codec_layer = codec_layer
        self.length = L
        self.number = len(self.meta)
        self.special_token_id = 2048
    
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.meta[idx]['location'][:-4]+'.npy')
        description = self.meta[idx]['main_caption']
        data = np.load(path, allow_pickle=True)
        data = data[:self.codec_layer, :]
        K, L = data.shape
        # print(K, L)
        # data = torch.LongTensor(np.pad(data,((0, 0), (self.length-L, 0)),'constant',constant_values=(0,0)))
        if L >= self.length:
            data = data[:, :self.length]
        else:
            data = torch.LongTensor(np.pad(data,((0, 0), (0, self.length-L)), 'constant', constant_values=(0, self.special_token_id)))
        data = to_delay(data)
        # data = torch.LongTensor(data)
        # mask = torch.ne(torch.LongTensor(data), 2048)
        # return seq[:, 0:499], seq[:, 1:500]
        mask = (data == 2048)
        return torch.LongTensor(data[:, :-1]), mask[0], torch.LongTensor(data[:, 1:]), description
    
    def __len__(self):
        return len(self.meta)

def temperature_sampling(logits, temperature, topk):
        # probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        logits = torch.Tensor(logits)
        probs = nn.Softmax(dim=0)(logits / temperature)
        probs = np.array(probs)
        if topk == 1:
            prediction = np.argmax(probs)
        else:
            sorted_index = np.argsort(probs)[::-1]
            candi_index = sorted_index[:topk]
            candi_probs = [probs[i] for i in candi_index]
            # normalize probs
            candi_probs /= sum(candi_probs)
            # choose by predicted probs
            prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return prediction


def train():
    print('setup...', end='\r')
    
    ########################################################################
    # training
    EPOCH = 200
    start_epoch = 1
    BATCH = opt.batch
    project_name = opt.project_name
    max_grad_norm = 1
    device = opt.device
    accumulation_step = opt.accumulation_step
    dataset_type = 'MusicBench'
    ################################################
    # model config setup
    model_config = {
        'layers':opt.layer_num,
        'vocab_size':2048+1,
        'd_model':1024,
        'drop_p':0.4,
        'd_state':opt.d_state,
        # add to parser
        'inner': opt.is_inner,
        'num_heads': 8,
        'self_atten_layers': [1, 2, 22, 23],
    }
    ################################################
    # optimizer config setup
    optim_config = {
        'optim_lr': 5e-6,
        'weight_decay':0.02,
        'betas': (0.9, 0.999),
    }
    ########################################################################
    
    # ckpts folder path
    os.makedirs('./ckpts', exist_ok=True)
    ckpt_folder = './ckpts/{}'.format(project_name)
    os.makedirs(ckpt_folder, exist_ok=True)
    
    # log path
    log_path = ckpt_folder
    
    # dataset
    metadata_path = '/mnt/gestalt/home/lonian/datasets/MusicBench/musicbench_train_simba.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
    train_data = MB_Dataset(metadata, root_path = '/mnt/gestalt/home/lonian/datasets/MusicBench/data_token')
    train_loader = DataLoader(dataset=train_data, batch_size = BATCH, shuffle=True, num_workers=4, pin_memory=True)
    
    # model and optimizer and scheduler
    print('Loading model...', end='\r')
    
    music_model = Text_Mmamba(
        layers = model_config['layers'],
        vocab_size = model_config['vocab_size'],
        d_model = model_config['d_model'],
        drop_p = model_config['drop_p'], 
        d_state = model_config['d_state'], 
        num_heads = model_config['num_heads'], 
        inner = model_config['inner'], 
        self_atten_layers = model_config['self_atten_layers'])
    
    music_model = music_model.to(device)
    
    # text encoder
    text_encoder_name = 'google/flan-t5-base'
    tokenizer = T5Tokenizer.from_pretrained(text_encoder_name)
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_name).train(mode=False)
    text_encoder = text_encoder.to(device)
    
    config = {}
    config['training'] = {
        'name': project_name,
        'dataset': dataset_type, 
        'epoch': EPOCH,
        'data_number': len(metadata),
        'batch': BATCH,
        'accumulation_step': accumulation_step,
        'model_size': cal_torch_model_params(music_model)
    }
    config['model'] = model_config

    # original optimizer and scheduler settings
    optimizer = AdamW(  music_model.parameters(), 
                        lr = optim_config['optim_lr'], 
                        weight_decay = optim_config['weight_decay'], 
                        betas = optim_config['betas']    )

    warm_up_iter = 10
    T_max = EPOCH	# 周期
    lr_max = 1e-1	# 最大值
    lr_min = 5e-4	# 最小值

    # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
    lambda0 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
    (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    
    config['optimizer'] = optim_config
    config['scheduler'] = {
        'warmup_duration': warm_up_iter,
        'T_max': EPOCH
    }
    ##########################################################################################
    
    with open(os.path.join(ckpt_folder, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    
    losses_list = []
    min_loss = 100
    losses=0
    logger = create_logger(log_path)
    
    # Check dataset
    '''
    logger.info('------Check dataset------')
    for x, y, text in train_loader:
        # x, y, text = batch
        batch = tokenizer(
            text, padding=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        with torch.no_grad():
            text_embedding = text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
        
        # dropout texts w/ 10%
        # dropout words w/ 5%
        attention_mask = random_zeroing(attention_mask)
        text_embedding_mask = (attention_mask == 1).to(device)
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.info("audio data (x) contains NaN or Inf")
        if torch.isnan(y).any() or torch.isinf(y).any():
            logger.info("audio data (y) contains NaN or Inf")
        if torch.isnan(text_embedding).any():
            logger.info("text data contains NaN or Inf")
    logger.info('------Check dataset done------')
    '''
    # logger = create_logger(log_path)
    logger.info('------Begin Training Model------')
    logger.info(config)
    logger.info(music_model)
    scheduler.step()
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, EPOCH+1):
        music_model.train()
        single_epoch = []
        iter_id = 1
        optimizer.zero_grad()
        # for idx, (x, y, text) in enumerate(train_loader):
            # print('>>>>>\tEPOCH: {}\t Loss: {}'.format(idx, losses), end='\r')
        for x, x_mask, y, text in tqdm(train_loader, ncols=120):
            x = x.to(device)
            y = y.to(device)
            x_mask = x_mask[:, :-1].to(device)
            # mask = mask.to(device)
            # process text
            batch = tokenizer(
                text, padding=True, return_tensors="pt"
            )
            input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

            with torch.set_grad_enabled(False):
                text_embedding = text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
            
            # dropout texts w/ 10%
            # dropout words w/ 5%
            attention_mask = random_zeroing(attention_mask)
            text_embedding_mask = (attention_mask == 1).to(device)
            # text_embedding = text_embedding.to(device)
            
            torch.cuda.set_device(x.device.index)
            output_logit = music_model(x, text_embedding, text_embedding_mask)
            # output_logit = torch.clamp(output_logit, min=1e-9, max=1e9)
            
            # check model output
            if torch.isnan(output_logit).any() or torch.isinf(output_logit).any():
                logger.info("Model output contains NaN or Inf")
                logger.info("NaN =",torch.isnan(output_logit).any(), "\nInf =", torch.isinf(output_logit).any())
                cc = input('===================================== PAUSE =====================================')
                v_n = []
                v_v = []
                v_g = []
                for name, parameter in music_model.named_parameters():
                    v_n.append(name)
                    v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
                    v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])
                for i in range(len(v_n)):
                    if np.max(v_v[i]).item() - np.min(v_v[i]).item() < 1e-6:
                        color = '*\t'
                    else:
                        color = '\t'
                    logger.info(color+'value %s: %.3e ~ %.3e' % (v_n[i], np.min(v_v[i]).item(), np.max(v_v[i]).item()))
                    logger.info(color+'grad  %s: %.3e ~ %.3e' % (v_n[i], np.min(v_g[i]).item(), np.max(v_g[i]).item()))
                cc = input('===================================== PAUSE =====================================')
                # logger.info(output_logit)
            
            # output_logit = out
            losses = 0
            for k in range(4):
                logits_k = output_logit[:, k, :, :].contiguous().view(-1, output_logit.size(-1))
                targets_k = y[:, k, :].contiguous().view(-1)
                # logits_mask = mask[:, k, 1:].contiguous().view(-1)
                loss = nn.CrossEntropyLoss(ignore_index=2048)(logits_k, targets_k)
                
                # check loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.info("Loss contains NaN or Inf")
                    # logger.info("NaN = "+str(torch.isnan(loss).any())+"\nInf ="+str(torch.isinf(loss).any()))
                    cc = input('===================================== PAUSE =====================================')
                    v_n = []
                    v_v = []
                    v_g = []
                    for name, parameter in music_model.named_parameters():
                        v_n.append(name)
                        v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
                        v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])
                    for i in range(len(v_n)):
                        if np.max(v_v[i]).item() - np.min(v_v[i]).item() < 1e-6:
                            color = '*\t'
                        else:
                            color = '\t'
                        logger.info(color+'value %s: %.3e ~ %.3e' % (v_n[i], np.min(v_v[i]).item(), np.max(v_v[i]).item()))
                        logger.info(color+'grad  %s: %.3e ~ %.3e' % (v_n[i], np.min(v_g[i]).item(), np.max(v_g[i]).item()))
                    cc = input('===================================== PAUSE =====================================')
                
                losses += loss
            
            losses = losses / (4*accumulation_step)
            losses.backward()
            
            '''
            check = float(losses)
            if math.isnan(check):
                logger.info('===============================NaN===========================================')
                logger.info("Before gradients")
                for name, param in grad_log:
                    # if param.grad is not None:
                    logger.info("Layer: {} | Grad max: {} | Grad min: {}".format(name, param.grad.max(), param.grad.min()))
                logger.info('===============================NaN===========================================')
                
                logger.info('===============================NaN===========================================')
                logger.info("Epoch: {}, Batch: {}, Loss: {}".format(epoch, idx, losses.item()))
                # logger.info("x: {}\ny: {}\nText: {}".format(x.to('cpu'), y.to('cpu'), text_embedding.to('cpu')))
                # logger.info(path)
                # logger.info(text)
        
                # 检查梯度
                for name, param in music_model.named_parameters():
                    if param.grad is not None:
                        logger.info("Layer: {} | Grad max: {} | Grad min: {}".format(name, param.grad.max(), param.grad.min()))
                logger.info('===============================NaN===========================================')
                a = input('===================================== PAUSE =====================================')
            else:
                # logger.info('NaN')
                logger.info("Epoch: {}, Batch: {}, Loss: {}".format(epoch, idx, losses.item()))
                grad_log = music_model.named_parameters()
                # logger.info(path)
                # logger.info(text)
                # for emb_idx, emb in enumerate(text_embedding.to('cpu')):
                #     logger.info("Text_{}: {}".format(emb_idx, emb))
                # a = input('===================================== GOOD PAUSE =====================================')
            '''
            
            if iter_id % accumulation_step == 0 or iter_id==len(train_loader):
                torch.nn.utils.clip_grad_norm_(music_model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            
            single_epoch.append(losses.to('cpu').mean().item())
            iter_id += 1
        
        single_epoch = np.array(single_epoch)
        losses_list.append(single_epoch.mean()*accumulation_step)
        
        logger.info('>>> Epoch: {} | Loss: {:.4f} | Lr: {}'.format(epoch, losses_list[-1], optimizer.param_groups[0]['lr']))
        scheduler.step()
        
        
        
        if epoch % 2 == 0:
            torch.save({'epoch': epoch,
                        'model': music_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss': losses_list[-1],
                        }, os.path.join(ckpt_folder, 'epoch_%03d.pkl'%epoch))
        
        if losses_list[-1] < min_loss:
            torch.save({'epoch': epoch,
                        'model': music_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss': losses_list[-1],
                        }, os.path.join(ckpt_folder, 'best.pkl'))
        
        np.save(os.path.join(ckpt_folder, 'training_loss'), np.array(losses_list))


def cont_train():
    print('CONTINUE setup...', end='\r')
    
    ########################################################################
    cont_epoch = opt.ckpt
    assert cont_epoch % 2 == 0
    # training
    EPOCH = 200
    start_epoch = 1
    project_name = opt.project_name
    device = opt.device
    
    max_grad_norm = 1
    dataset_type = 'MusicBench'
    ################################################
    with open('/mnt/gestalt/home/lonian/mamba/model/ckpts/{}/config.json'.format(project_name)) as f:
        config = json.load(f)
    # training config setup
    train_config = config['training']
    BATCH = train_config['batch']
    accumulation_step = train_config['accumulation_step']
    ################################################
    # model config setup
    model_config = config['model']
    ################################################
    # optimizer config setup
    optim_config = config['optimizer']
    ########################################################################
    
    # ckpts folder path
    ckpt_folder = '/mnt/gestalt/home/lonian/mamba/model/ckpts/{}'.format(project_name)
    os.makedirs(ckpt_folder, exist_ok=True)
    
    # dataset
    metadata_path = '/mnt/gestalt/home/lonian/datasets/MusicBench/musicbench_train_simba.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
    train_data = MB_Dataset(metadata, root_path = '/mnt/gestalt/home/lonian/datasets/MusicBench/data_token')
    train_loader = DataLoader(dataset=train_data, batch_size = BATCH, shuffle=True, num_workers=4, pin_memory=True)
    
    # model and optimizer and scheduler
    print('Loading model...', end='\r')
    
    music_model = Text_Mmamba(
        layers = model_config['layers'],
        vocab_size = model_config['vocab_size'],
        d_model = model_config['d_model'],
        drop_p = model_config['drop_p'], 
        d_state = model_config['d_state'], 
        num_heads = model_config['num_heads'], 
        inner = model_config['inner'], 
        self_atten_layers = model_config['self_atten_layers'])
    
    music_model = music_model.to(device)
    # original optimizer and scheduler settings
    optimizer = AdamW(  music_model.parameters(), 
                        lr = optim_config['optim_lr'], 
                        weight_decay = optim_config['weight_decay'], 
                        betas = optim_config['betas']    )
    
    warm_up_iter = 10
    T_max = EPOCH	# 周期
    lr_max = 1e-1	# 最大值
    lr_min = 5e-4	# 最小值

    # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
    lambda0 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
    (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    
    ##########################################################################################
    # text encoder
    text_encoder_name = 'google/flan-t5-base'
    tokenizer = T5Tokenizer.from_pretrained(text_encoder_name)
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_name).train(mode=False)
    text_encoder = text_encoder.to(device)
    
    checkpoints_path = '/mnt/gestalt/home/lonian/mamba/model/ckpts/{}/epoch_{:03d}.pkl'.format(project_name, cont_epoch)
    if os.path.isfile(checkpoints_path):
        checkpoint = torch.load(checkpoints_path, map_location=device)
    else:
        os._exit()
    start_epoch = checkpoint['epoch'] + 1
    music_model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    losses_list = list(np.load('/mnt/gestalt/home/lonian/mamba/model/ckpts/{}/training_loss.npy'.format(project_name)))
    losses_list = losses_list[:cont_epoch]
    assert len(losses_list) == cont_epoch
    # print(len(losses_list), min(losses_list))
    min_loss = min(losses_list)
    
    # log path
    log_path = ckpt_folder
    logger = create_logger(log_path)
    logger.info('------Continue Training Model------')
    logger.info(config)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, EPOCH+1):
        music_model.train()
        single_epoch = []
        iter_id = 1
        optimizer.zero_grad()
        # for idx, (x, y, text) in enumerate(train_loader):
            # print('>>>>>\tEPOCH: {}\t Loss: {}'.format(idx, losses), end='\r')
        for x, x_mask, y, text in tqdm(train_loader, ncols=120):
            x = x.to(device)
            y = y.to(device)
            x_mask = x_mask[:, :-1].to(device)
            # mask = mask.to(device)
            # process text
            batch = tokenizer(
                text, padding=True, return_tensors="pt"
            )
            input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

            with torch.set_grad_enabled(False):
                text_embedding = text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
            
            # dropout texts w/ 10%
            # dropout words w/ 5%
            attention_mask = random_zeroing(attention_mask)
            text_embedding_mask = (attention_mask == 1).to(device)
            # text_embedding = text_embedding.to(device)
            
            torch.cuda.set_device(x.device.index)
            output_logit = music_model(x, text_embedding, text_embedding_mask)
            # output_logit = torch.clamp(output_logit, min=1e-9, max=1e9)
            
            '''
            # check model output
            if torch.isnan(output_logit).any() or torch.isinf(output_logit).any():
                logger.info("Model output contains NaN or Inf")
                # logger.info("NaN =",torch.isnan(output_logit).any(), "\nInf =", torch.isinf(output_logit).any())
                cc = input('===================================== PAUSE =====================================')
                v_n = []
                v_v = []
                v_g = []
                for name, parameter in music_model.named_parameters():
                    v_n.append(name)
                    v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
                    v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])
                for i in range(len(v_n)):
                    if np.max(v_v[i]).item() - np.min(v_v[i]).item() < 1e-6:
                        color = '*\t'
                    else:
                        color = '\t'
                    logger.info(color+'value %s: %.3e ~ %.3e' % (v_n[i], np.min(v_v[i]).item(), np.max(v_v[i]).item()))
                    logger.info(color+'grad  %s: %.3e ~ %.3e' % (v_n[i], np.min(v_g[i]).item(), np.max(v_g[i]).item()))
                cc = input('===================================== PAUSE =====================================')
                # logger.info(output_logit)
            '''
            
            # output_logit = out
            losses = 0
            for k in range(4):
                logits_k = output_logit[:, k, :, :].contiguous().view(-1, output_logit.size(-1))
                targets_k = y[:, k, :].contiguous().view(-1)
                # logits_mask = mask[:, k, 1:].contiguous().view(-1)
                loss = nn.CrossEntropyLoss(ignore_index=2048)(logits_k, targets_k)
                
                '''
                # check loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.info("Loss contains NaN or Inf")
                    # logger.info("NaN = "+str(torch.isnan(loss).any())+"\nInf ="+str(torch.isinf(loss).any()))
                    cc = input('===================================== PAUSE =====================================')
                    v_n = []
                    v_v = []
                    v_g = []
                    for name, parameter in music_model.named_parameters():
                        v_n.append(name)
                        v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
                        v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])
                    for i in range(len(v_n)):
                        if np.max(v_v[i]).item() - np.min(v_v[i]).item() < 1e-6:
                            color = '*\t'
                        else:
                            color = '\t'
                        logger.info(color+'value %s: %.3e ~ %.3e' % (v_n[i], np.min(v_v[i]).item(), np.max(v_v[i]).item()))
                        logger.info(color+'grad  %s: %.3e ~ %.3e' % (v_n[i], np.min(v_g[i]).item(), np.max(v_g[i]).item()))
                    cc = input('===================================== PAUSE =====================================')
                '''
                
                losses += loss
            
            losses = losses / (4*accumulation_step)
            losses.backward()
            
            if iter_id % accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(music_model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            
            single_epoch.append(losses.to('cpu').mean().item())
            iter_id += 1
        
        
        
        single_epoch = np.array(single_epoch)
        losses_list.append(single_epoch.mean()*accumulation_step)
        # print('')
        logger.info('>>> Epoch: {} | Loss: {:.4f}'.format(epoch, losses_list[-1]))
        scheduler.step()
        
        
        if epoch % 2 == 0:
            torch.save({'epoch': epoch,
                        'model': music_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss': losses_list[-1],
                        }, os.path.join(ckpt_folder, 'epoch_%03d.pkl'%epoch))
        
        if losses_list[-1] < min_loss:
            torch.save({'epoch': epoch,
                        'model': music_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss': losses_list[-1],
                        }, os.path.join(ckpt_folder, 'best.pkl'))
        
        np.save(os.path.join(ckpt_folder, 'training_loss'), np.array(losses_list))

r'''
def test():
    ################################################
    # sampling setup
    device = 'cuda:2'
    temperature = 1.2
    topk = 200
    topp = 0.95
    prompt_length = 100
    model_path = '/mnt/gestalt/home/lonian/mamba/model/ckpts/v10/epoch_180.pkl'
    ################################################
    # model config setup
    model_config = MambaConfig()
    model_config.n_layer = 48
    model_config.attn_layer_idx = [11, 23, 35, 47]
    model_config.attn_cfg = {'num_heads': 16}
    model_config.vocab_size = 2048
    model_config.d_model = 1024
    model_config.ssm_cfg = {"layer": "Mamba2", "d_state":512}
    ################################################
    os.makedirs('./results', exist_ok=True)
    # model_path = '/mnt/gestalt/home/lonian/mamba/model/ckpts/v9/best.pkl'
    idx = len(glob('/mnt/gestalt/home/lonian/mamba/model/results/*.npy'))+1
    
    with torch.no_grad():
        # load prompt
        prompt_id = [2]
        B = len(prompt_id)
        prompt = []
        for i in prompt_id:
            a = np.load('/mnt/gestalt/home/lonian/datasets/mamba_test_token/{}.npy'.format(i), allow_pickle=True)
            prompt.append(a)
        # print(prompt)
        prompt = np.array(prompt)
        prompt_seq = prompt[:, :, :prompt_length]
        
        # prompt = np.load('/mnt/gestalt/home/lonian/datasets/mamba_test_token/2.npy', allow_pickle=True)
        # prompt_seq = prompt[:, :prompt_length]
        
        # load model
        checkpoint = torch.load(model_path, map_location={'cuda:2': device})
        # checkpoint = torch.load(model_path)
        music_model = MusicMambaLMHeadModel(model_config)
        music_model = music_model.to(device)
        music_model.load_state_dict(checkpoint['model'])
        music_model.eval()
        
        input_seq = torch.LongTensor(prompt_seq).to(device)
        torch.cuda.set_device(input_seq.device.index)
        out = music_model.generate( input_ids=input_seq, 
                                    max_length=1500, 
                                    temperature=temperature, 
                                    top_k=topk, 
                                    top_p = topp    )
    
    for b in range(B):
        np.save('/mnt/gestalt/home/lonian/mamba/model/results/{}.npy'.format(idx), out[b])
        idx+=1
    print('\n', prompt_seq.shape)
    pass
'''

def evaluate():
    pass

def main():
    if opt.is_continue:
        cont_train()
    else:
        train()


if __name__ == '__main__':
    main()