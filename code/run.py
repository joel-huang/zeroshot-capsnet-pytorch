import os
from random import *
import time

from dataset import IntentDataset, batch_function
from model import CapsuleNetwork

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import math

from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

a = Random()
a.seed(1)
torch.cuda.manual_seed_all(17)
device = torch.device("cuda:0")

def setting(train_set, test_set, embedding):
    vocab_size, word_emb_size = embedding.shape
    max_time = sorted(train_set, reverse=True, key=lambda x: x['length'])[0]
    train_num = len(train_set)
    test_num = len(test_set)
    s_cnum = len(train_set.class_list)
    u_cnum = len(test_set.class_list)
    config = {}
    config['keep_prob'] = 0.5 # embedding dropout keep rate
    config['hidden_size'] = 64 # embedding vector size
    config['batch_size'] = 32 # vocab size of word vectors
    config['vocab_size'] = vocab_size # vocab size (10895) after subtracting padding
    config['num_epochs'] = 5 # number of epochs
    config['max_time'] = max_time
    config['sample_num'] = train_num # sample number of training data
    config['test_num'] = test_num # number of test data
    config['s_cnum'] = s_cnum # seen class num
    config['u_cnum'] = u_cnum # unseen class num
    config['word_emb_size'] = word_emb_size # embedding size of word vectors (300)
    config['d_a'] = 20 # self-attention weight hidden units number
    config['output_atoms'] = 10 # capsule output atoms
    config['r'] = 3 # self-attention weight hops
    config['num_routing'] = 2 # capsule routing num
    config['alpha'] = 0.0001 # coefficient of self-attention loss
    config['margin'] = 1.0 # ranking loss margin
    config['learning_rate'] = 0.00005
    config['sim_scale'] = 4 # sim scale
    config['nlayers'] = 2 # default for bilstm
    config['ckpt_dir'] = './saved_models/' # check point dir
    return config

def get_sim(train_set, test_set):
    """
    get unseen and seen categories similarity.
    """
    seen = normalize(torch.stack(list(train_set.class_w2v.values())))
    unseen = normalize(torch.stack(list(test_set.class_w2v.values())))
    sim = utils.compute_label_sim(unseen, seen, config['sim_scale'])
    return torch.from_numpy(sim)

def _squash(input_tensor):
    norm = torch.norm(input_tensor, dim=2, keepdim=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (0.5 + norm_squared))

def update_unseen_routing(votes, config, num_routing=3):
    votes_t_shape = [3, 0, 1, 2]
    r_t_shape = [1, 2, 3, 0]
    votes_trans = votes.permute(votes_t_shape)
    num_dims = 4
    input_dim = config['r']
    output_dim = config['u_cnum']
    input_shape = votes.shape
    logit_shape = np.stack([input_shape[0], input_dim, output_dim])
    logits = torch.zeros(logit_shape[0], logit_shape[1], logit_shape[2]).cuda()
    activations = []

    for iteration in range(num_routing):
        route = F.softmax(logits, dim=2).cuda()
        preactivate_unrolled = route * votes_trans
        preact_trans = preactivate_unrolled.permute(r_t_shape)

        # delete bias to fit for unseen classes
        preactivate = torch.sum(preact_trans, dim=1)
        activation = _squash(preactivate)
        # activations = activations.write(i, activation)
        activations.append(activation)
        # distances: [batch, input_dim, output_dim]
        act_3d = torch.unsqueeze(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = act_3d.repeat(tile_shape)
        distances = torch.sum(votes * act_replicated, dim=3)
        logits = logits + distances

    return activations[num_routing-1], route

data_prefix = '../data/nlu_data/'
w2v_path = data_prefix + 'wiki.en.vec'
training_data_path = data_prefix + 'train_shuffle.txt'
test_data_path = data_prefix + 'test.txt'

seen_classes = ['music', 'search', 'movie', 'weather', 'restaurant']
unseen_classes = ['playlist', 'book']

train_set = IntentDataset(seen_classes, w2v_path, training_data_path)
test_set = IntentDataset(unseen_classes, w2v_path, test_data_path)

embedding = train_set.embedding
categorical = train_set.categorical
config = setting(train_set, test_set, embedding)
similarity = get_sim(train_set, test_set).to(device)

train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True,
                          collate_fn=batch_function, num_workers=4)
test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=True,
                         collate_fn=batch_function, num_workers=4)

model = CapsuleNetwork(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
#scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True, threshold=0.001)

if os.path.exists(config['ckpt_dir'] + 'best_model.pt'):
    print("Restoring weights from previously trained rnn model.")
    model.load_state_dict(torch.load(config['ckpt_dir'] + 'best_model.pt' ))
else:
    print('Initializing Variables')
    if not os.path.exists(config['ckpt_dir']):
        os.mkdir(config['ckpt_dir'])

def train(epoch, train_loader, config, model, embedding, train_time):
    model.train()
    avg_acc = 0
    avg_loss = 0
    start_time = time.time()
    
    for idx, batch in enumerate(train_loader):
        input = batch.sentences_w2v.cuda()
        lengths = batch.lengths
        target = batch.label_onehot.cuda()
        label_w2v = batch.label_w2v
        
        batch_size = len(input)
        hc = (Variable(torch.zeros(4, input.shape[0], config['hidden_size'])).cuda(),
              Variable(torch.zeros(4, input.shape[0], config['hidden_size'])).cuda())

        output = model(input, lengths, embedding.cuda(), hc)
        loss = model.loss(target.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        clone_logits = model.logits.detach().clone()
        pred = torch.argmax(clone_logits, 1).cpu()
        acc = accuracy_score(categorical(target.cpu()), pred)
        print("Epoch: {}\t| Batch: {:03d}/{}\t| Batch Loss: {}\t| Acc: {}%".format(
                epoch, (idx+1), len(train_loader), round(loss.item(), 4), round(acc * 100., 2)))
        avg_loss += loss.item()
        avg_acc += acc

    epoch_time = time.time() - start_time
    train_time += epoch_time
    avg_loss /= len(train_loader)
    avg_acc /= len(train_loader)
    
    print("Epoch: {}\t| Average Loss: {}\t| Average Acc: {}%\t| Train Time: {}s".format(
          epoch, round(avg_loss, 4), round(avg_acc * 100., 2), round(train_time, 2)))

    return avg_loss, avg_acc, train_time

def test(epoch, test_loader, config, model, embedding, similarity):
    # zero-shot testing state
    # seen votes shape (110, 2, 34, 10)
    # get unseen and seen categories similarity
    # sim shape (8, 34)
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            input = batch.sentences_w2v.cuda()
            lengths = batch.lengths
            target = batch.label_onehot.long().cuda()
            label_w2v = batch.label_w2v
        
            batch_size = len(input)
            hc = (Variable(torch.zeros(4, input.shape[0], config['hidden_size'])).cuda(),
                  Variable(torch.zeros(4, input.shape[0], config['hidden_size'])).cuda())

            output = model.forward(input, lengths, embedding.cuda(), hc)
            attentions, seen_logits, seen_votes, seen_weights_c = model.attention, model.logits, \
                                                                  model.votes, model.weights_c
            sim = similarity.unsqueeze(0)
            sim = sim.repeat(seen_votes.shape[1], 1, 1).unsqueeze(0)
            sim = sim.repeat(seen_votes.shape[0], 1, 1, 1)
            seen_weights_c = seen_weights_c.unsqueeze(-1)
            seen_weights_c = seen_weights_c.repeat(1, 1, 1, config['output_atoms'])
            mul = seen_votes * seen_weights_c

            # compute unseen features
            # unseen votes shape (110, 2, 8, 10)
            unseen_votes = torch.matmul(sim, mul)

            # routing unseen classes
            u_activations, u_weights_c = update_unseen_routing(unseen_votes, config, 3)
            unseen_logits = torch.norm(u_activations, dim=-1)
            batch_pred = torch.argmax(unseen_logits, dim=1).unsqueeze(1).cuda()
            
            if idx == 0:
                total_pred = batch_pred
                total_target = target
            else:
                total_pred = torch.cat((total_pred.cuda(), batch_pred))
                total_target = torch.cat((total_target.cuda(), target))
                
    print ("           zero-shot intent detection test set performance        ")
    cpu_target = categorical(total_target.cpu())
    cpu_pred = total_pred.flatten().cpu()
    acc = accuracy_score(cpu_target, cpu_pred)
    print (classification_report(cpu_target, cpu_pred, digits=4))
            
    test_time = time.time() - start_time      
    return acc, test_time

best_acc = 0
train_time, test_time = 0, 0

for epoch in range(1, config['num_epochs'] + 1):
    train_loss, train_acc, train_time = train(epoch, train_loader, config, model, embedding, train_time)
    test_acc, test_time = test(epoch, test_loader, config, model, embedding, similarity)
    #scheduler.step(test_acc)
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), config['ckpt_dir'] + 'best_model.pt')

    print("test_acc", test_acc)
    print("best_acc", best_acc)
    print("Testing time", round(test_time, 4))

print("Overall training time", train_time)
print("Overall testing time", test_time)