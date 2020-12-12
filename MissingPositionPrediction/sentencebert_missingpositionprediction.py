import os
import itertools
import argparse
import random
import math
import time

import numpy as np
import pandas as pd

import spacy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

# from torchtext.datasets import TranslationDataset, Multi30k
# from torchtext.data import Field, BucketIterator

from transformers import *
from sentence_transformers import SentenceTransformer

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
from collections import namedtuple #, deque 


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

# ------------
# Data loading
# ------------

class ROCStoriesDataset_with_missing(Dataset):
    def __init__(self, data_path=""):
        assert os.path.isfile(data_path)

        self.df = pd.read_csv(data_path)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):       
        row = self.df.iloc[idx].values
        
        story_lines = row[0:4]
        missing_id = row[4]
        missing_sentence = row[5:6]
        
        return story_lines, missing_sentence, missing_id
    

class ROCStoriesDataset_random_missing(Dataset):
    def __init__(self, data_path=""):
        assert os.path.isfile(data_path)

        self.df = pd.read_csv(data_path)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):       
        row = self.df.iloc[idx].values
        
        story_lines = row[0:5]
        
        missing_id = np.random.randint(low=0, high=5) 
        
        missing_sentence = np.array([story_lines[missing_id]], dtype=object)
        remain_sentences = np.delete(story_lines, missing_id)
        
        return remain_sentences, missing_sentence, missing_id
    

# --------------------------
# Encoding and preprocessing
# --------------------------

def fit_to_block_size(sequence, block_size, pad_token):
    """ Adapt the source and target sequences' lengths to the block size.
    If the sequence is shorter than the block size we pad it with -1 ids
    which correspond to padding tokens.
    """
    if len(sequence) > block_size:
        return sequence[:block_size]
    else:
        sequence.extend([pad_token] * (block_size - len(sequence)))
        return sequence


def build_lm_labels(sequence, pad_token):
    """ Padding token, encoded as 0, are represented by the value -1 so they
    are not taken into account in the loss computation. """
    padded = sequence.clone()
    padded[padded == pad_token] = -1
    return padded


def build_mask(sequence, pad_token):
    """ Builds the mask. The attention mechanism will only attend to positions
    with value 1. """
    mask = torch.ones_like(sequence)
    idx_pad_tokens = sequence == pad_token
    mask[idx_pad_tokens] = 0
    return mask


def encode_for_storycompletion(story_lines, missing_sentence, tokenizer):
    """ Encode the story lines and missing sentence, and join them
    as specified in [1] by using `[SEP] [CLS]` tokens to separate
    sentences.
    """
    story_lines_token_ids = [
        tokenizer.encode(line, add_special_tokens=True)
        for line in story_lines
    ]
    missing_sentence_token_ids = [
        tokenizer.encode(line, add_special_tokens=True)
        for line in missing_sentence
    ]

    story_token_ids = [
        token for sentence in story_lines_token_ids for token in sentence
    ]
    missing_sentence_token_ids = [
        token for sentence in missing_sentence_token_ids for token in sentence
    ]

    return story_token_ids, missing_sentence_token_ids, story_lines_token_ids


def compute_token_type_ids(batch, separator_token_id):
    """ Segment embeddings as described in [1]
    The values {0,1} were found in the repository [2].
    Attributes:
        batch: torch.Tensor, size [batch_size, block_size]
            Batch of input.
        separator_token_id: int
            The value of the token that separates the segments.
    [1] Liu, Yang, and Mirella Lapata. "Text summarization with pretrained encoders."
        arXiv preprint arXiv:1908.08345 (2019).
    [2] https://github.com/nlpyang/PreSumm (/src/prepro/data_builder.py, commit fac1217)
    """
    batch_embeddings = []
    for sequence in batch:
        sentence_num = 0
        embeddings = []
        for s in sequence:
            if s == separator_token_id:
                sentence_num += 1
            embeddings.append(sentence_num % 2)
        batch_embeddings.append(embeddings)
    return torch.tensor(batch_embeddings)


# ----------------
# LOAD the dataset
# ----------------

Batch = namedtuple(
    "Batch", ["batch_size", "src", "mask_src", "missing_ids", "trg", "mask_trg", "tgt_str"]
)


def build_data_iterator(args, tokenizer):
    dataset = load_and_cache_examples(args, tokenizer)
    sampler = SequentialSampler(dataset)
    collate_fn = lambda data: collate(data, tokenizer, block_size=512, device=args.device)
    iterator = DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn,
    )

    return iterator


def load_and_cache_examples(args, tokenizer):
    dataset = ROCStoriesDataset(args.data_path)
    return dataset


def collate(data, tokenizer, block_size, device):
    """ Collate formats the data passed to the data loader.
    In particular we tokenize the data batch after batch to avoid keeping them
    all in memory. We output the data as a namedtuple to fit the original BertAbs's
    API.
    """
    story_lines = [story_lines for story_lines, _, _ in data]
    missing_ids = torch.tensor([ids for _, _, ids in data])
    missing_sentences = [" ".join(missing_sentence) for _, missing_sentence, _ in data]

    encoded_text = [
        encode_for_storycompletion(story_lines, missing_sentence, tokenizer) 
        for story_lines, missing_sentence, _ in data
    ]
    
    
    encoded_stories = torch.tensor(    [
        [fit_to_block_size(line, block_size, tokenizer.pad_token_id) for line in story]
        for _, _, story in encoded_text
    ])  

    encoded_missing_sentences = torch.tensor([
        fit_to_block_size(missing_sentence, block_size, tokenizer.pad_token_id)
        for _, missing_sentence, _ in encoded_text
    ])  
    
    
    """
    encoded_stories = torch.tensor(
        [
            fit_to_block_size(line, block_size, tokenizer.pad_token_id)
            for _, _, story in encoded_text for line in story 
        ]
    )
    """
    # encoder_token_type_ids = compute_token_type_ids(encoded_stories, tokenizer.cls_token_id)
    encoder_mask = build_mask(encoded_stories, tokenizer.pad_token_id)
    decoder_mask = build_mask(encoded_missing_sentences, tokenizer.pad_token_id)

    batch = Batch(
        batch_size=len(encoded_stories),
        src=story_lines,
        # segs=encoder_token_type_ids.to(device),
        mask_src=encoder_mask.to(device),
        missing_ids = missing_ids.to(device),
        trg=encoded_missing_sentences.to(device),
        mask_trg=decoder_mask.to(device),
        tgt_str=missing_sentences,
    )

    return batch


# -----
# Model
# -----
    
class GRUContextEncoder(nn.Module):
    def __init__(self, input_size=768, hidden_size=256):
        super().__init__()
        
        self.rnn = nn.GRU(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.linear.weight, 0.0, 0.01)
        self.bn = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        # x = [batch size, seq len (num sentence), hidden size]
        
        # trans_x = [seq len (num sentence), batch size, hidden size]
        trans_x = x.transpose(0, 1)
        
        # h = [batch size, hidden size]
        h = self.rnn(trans_x)[1][-1]

        h = self.linear(h)
        h = F.relu(self.bn(h))
        
        return h

    
class PoolContextEncoder(nn.Module):
    def __init__(self, input_size=768, hidden_size=256):
        super().__init__()
        
        self.linear = nn.Linear(input_size, hidden_size)
        nn.init.normal_(self.linear.weight, 0.0, 0.01)
        self.bn = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        # x = [batch size, seq len (num sentence), hidden size]
        
        # trans_x = [seq len (num sentence), batch size, hidden size]
        trans_x = x.transpose(0, 1)
        
        # h = [batch size, hidden size]

        ## max pooling ##
        h = torch.max(trans_x, 0)[0]        
        h = self.linear(h)
        h = F.relu(self.bn(h))
        
        return h
    
    
class MissingPisitionPredictionModel(nn.Module):
    def __init__(self, SentenceEncoder, device, ContextEncoder="GRUContextEncoder"):
        super().__init__()
        
        self.sentence_encoder = SentenceEncoder
                
        # self.context_encoder = GRUContextEncoder(input_size=768, hidden_size=256)

        # Context Encoder
        if ContextEncoder == "GRUContextEncoder":
            self.context_encoder = GRUContextEncoder(input_size=768, hidden_size=256)
        elif ContextEncoder == "PoolContextEncoder":
            self.context_encoder = PoolContextEncoder(input_size=768, hidden_size=256)
        
        self.fc = nn.Linear(256, 5)

        self.device = device
        
        
    def forward(self, story):
        
        batch_size = len(story)
        
        all_sentences_in_batch = list(itertools.chain.from_iterable(story))
        embeddings = self.sentence_encoder.encode(all_sentences_in_batch, show_progress_bar=False)
        embeddings = np.stack(embeddings, axis=0)
        embeddings = embeddings.reshape(batch_size, 4, -1)
        
        # embeddings_tensor = [batch size, num sentences, feature]
        embeddings_tensor = torch.tensor(embeddings).to(self.device)
        
        # [num sentences, batch size, feature]
        # embeddings_tensor = embeddings_tensor.transpose(0, 1)
        
        # context = [batch size, feature]
        context = self.context_encoder(embeddings_tensor)

        outputs = self.fc(context)
        
        return outputs
    

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    epoch_acc = 0
    
    for i, batch in enumerate(tqdm(iterator, desc="Iteration")):        
        optimizer.zero_grad()

        batch_size = batch.batch_size
        story = batch.src
        cls = batch.missing_ids     
        
        output = model(story)
                
        loss = criterion(output, cls)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += (output.argmax(1) == cls).sum().item() / (batch_size + .0)
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator, desc="Iteration")):        

            batch_size = batch.batch_size
            story = batch.src
            cls = batch.missing_ids      

            output = model(story)

            loss = criterion(output, cls)

            epoch_loss += loss.item()
            epoch_acc += (output.argmax(1) == cls).sum().item() / (batch_size + .0)

    return epoch_loss / len(iterator), epoch_acc / len(iterator)           


def for_heatmap(model, iterator):
    model.eval()
    
    acc_heatmap = np.zeros((5, 5))
    cls_count = np.zeros(5)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator, desc="Iteration")):
            
            batch_size = batch.batch_size
            story = batch.src
            cls = batch.missing_ids
                        
            output = model(story)
            predicted = output.argmax(1)
            
            cls = cls.to("cpu").numpy()
            predicted = predicted.to("cpu").numpy()
            
            for e, c in zip(predicted, cls):
                acc_heatmap[e][c] += 1
                cls_count[c] += 1
            
    for i, cc in enumerate(cls_count):
        acc_heatmap[:][i] /= cc
            
    return acc_heatmap, cls_count


def show_result(model, iterator):
    model.eval()

    missing_sentence = "____________________."
    
    result_to_show = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator, desc="Iteration")):                
            batch_size = batch.batch_size
            story = batch.src
            cls = batch.missing_ids
            original_sentence = batch.tgt_str

            output = model(story)
            predicted = output.argmax(1)

            cls = cls.to("cpu").numpy()
            predicted = predicted.to("cpu").numpy()

            for i in range(batch_size):
                input_story = " ".join(story[i])
                predicted_missing_story = " ".join(np.insert(story[i], predicted[i], missing_sentence))
                gt_missing_story = " ".join(np.insert(story[i], cls[i], missing_sentence))
                gt_story = " ".join(np.insert(story[i], cls[i], original_sentence[i]))

                result_to_show.append([input_story, predicted[i], predicted_missing_story, cls[i], gt_missing_story, gt_story])
      
    show_result_df = pd.DataFrame(result_to_show, 
                                  columns=["input", "pred_missing_id (0_indexed)", "pred_missing_story", 
                                           "gt_missing_id (0_indexed)", "gt_missing_story", "gt_story"])
    
    return show_result_df    
    

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default="./", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of iterations to train')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Minibatch size')
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")    

    parser.add_argument('--context-encoder', '-ce', type=str, default='GRUContextEncoder',
                        help='type of context encoder')    
    
    args = parser.parse_args()
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set seed
    set_seed(args)    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = args.batch_size
    block_size = 32
    N_EPOCHS = args.epochs
    CLIP = 5    
    
    train_dataset = ROCStoriesDataset_random_missing(data_path = "../data/rocstories_completion_train.csv")
    val_dataset = ROCStoriesDataset_with_missing(data_path = "../data/rocstories_completion_dev.csv")
    
    sentbertmodel = SentenceTransformer('bert-base-nli-mean-tokens')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            
    # --- model --- 
    model = MissingPisitionPredictionModel(SentenceEncoder=sentbertmodel, device=device, ContextEncoder=args.context_encoder).to(device)
    
    # --- DataLoader ---
    collate_fn = lambda data: collate(data, tokenizer, block_size=block_size, device=device)
    train_iterator = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size, collate_fn=collate_fn,
    )    
    valid_iterator = DataLoader(
        val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size, collate_fn=collate_fn,
    )    
    
    TRG_PAD_IDX = tokenizer.pad_token_id
    START_ID = tokenizer.cls_token_id
    mpe_criterion = nn.CrossEntropyLoss()
    
    best_valid_loss = float('inf')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    tb_writer = SummaryWriter()

    for epoch in trange(N_EPOCHS, desc="Epoch"):

        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, mpe_criterion, CLIP)
        valid_loss, valid_acc = evaluate(model, valid_iterator, mpe_criterion)        

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'sentbert-positionestimation_model.pt'))

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_acc * 100:.1f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Accuracy: {valid_acc * 100:.1f}%')
        
        # tensorboard
        tb_writer.add_scalar('train_loss', train_loss, epoch+1)
        tb_writer.add_scalar('train_acc', train_acc, epoch+1)        
        tb_writer.add_scalar('valid_loss', valid_loss, epoch+1)
        tb_writer.add_scalar('valid_acc', valid_acc, epoch+1)        

    tb_writer.close()