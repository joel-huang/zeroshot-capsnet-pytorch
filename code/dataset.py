from gensim.models.keyedvectors import KeyedVectors
import tool

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class IntentDataset(Dataset):
    def __init__(self, class_list, w2v_path, data_path): 
        self.class_list = class_list
        self.w2v = self.load_w2v(w2v_path)
        self.class_indices, self.class_w2v = self.process_labels(class_list)
        self.embedding = self.load_embedding()
        self.corpus = self.load_data(data_path)
        
    def __getitem__(self, idx):
        """
        Each sample is a dictionary with keys
        {sentence (list of w2v indices), label_onehot, label_w2v}.
        """
        return self.corpus[idx]
    
    def __len__(self):
        return len(self.corpus)

    def load_w2v(self, model_path):
        """ 
        load w2v model
        input: model filepath
        returns: w2v model
        """
        w2v = KeyedVectors.load_word2vec_format(
                model_path, binary=False)
        return w2v

    def process_labels(self, class_list):
        """ 
        pre-process class labels
        input: list of string labels
        returns: two dicts, {labels-to-indices} and {labels-to-w2v}
        """
        label_indices = {}
        label_w2v = {}
        for k in range(len(class_list)):
            
            label = class_list[k]
            
            # Account for classes
            # with multiple words.
            label_list = label.split(' ')
            
            for word in label_list:
                if word not in self.w2v.vocab.keys():
                    raise Exception("{} not in W2V model".format(word))

            # Compute the sum of the
            # constituent words in the label
            vector_sum = 0
            for word in label_list:
                vector_sum += torch.Tensor(self.w2v[word])
            
            label_w2v[label] = vector_sum
            label_indices[label] = k
            
        return label_indices, label_w2v

    def load_data(self, file_path):
        """
        Loads samples into a list. Each sample is a dictionary with keys
        {sentence (list of w2v indices), label_onehot, label_w2v}.
        
        input: text file path
        returns: dataset, a list of dicts.
        """
        
        dataset = []
        
        for line in open(file_path):
            arr = line.strip().split('\t')
            label = [w for w in arr[0].split(' ')]
            sentence = [w for w in arr[1].split(' ')]
            cname = ' '.join(label)
            
            # The line is useless if the class is
            # not in the class dictionary.
            if cname not in self.class_list:
                raise Exception("{} not in class list.".format(cname))
            
            # Build the sample dictionary.
            sample = {}
            sample['sentence_w2v'] = []
            
            for word in sentence:
                if word not in self.w2v.vocab.keys():
                    continue # ignore sentence
                    
                # In the loading embedding (see self.load_embedding()), we
                # stack one additional layer of zeros in front to handle padding.
                # Thus here we append the embedding index plus one.
                sample['sentence_w2v'].append(torch.Tensor([self.w2v.vocab[word].index + 1]))

            sample['length'] = len(sample['sentence_w2v'])
            sample['label_onehot'] = self.onehot(self.class_indices[cname])
            sample['label_w2v'] = self.class_w2v[cname]
            dataset.append(sample)
        
        return dataset
    
    def categorical(self, onehot):
        return torch.argmax(onehot, dim=1)
    
    def onehot(self, idx):
        onehot = torch.zeros(len(self.class_indices))
        onehot[idx] = 1.
        return onehot

    def load_embedding(self):
        # load normalized word embeddings
        embedding = self.w2v.syn0
        norm_embedding = tool.norm_matrix(embedding)
        
        # Stack one layer of zeros on the embedding
        # to handle padding. So the total length of
        # the embedding increases by one.
        # See: https://datascience.stackexchange.com/questions/32345/initial-embeddings-for-unknown-padding
        # See: https://discuss.pytorch.org/t/padding-zeros-in-nn-embedding-while-using-pre-train-word-vectors/8443/4
        
        emb = torch.from_numpy(norm_embedding)
        zeros = torch.zeros(1, emb.shape[1])
        pad_enabled_embedding = torch.cat((zeros, emb))
        return pad_enabled_embedding
    
class IntentBatch:
    def __init__(self, batch):
        batch.sort(reverse=True, key=lambda x: x['length'])
        batch = list(zip(*map(dict.values, batch)))
        
        sentences_w2v = [torch.LongTensor(x) for x in batch[0]]
        lengths = torch.Tensor(batch[1]).long() # cpu
        label_onehot = torch.stack(batch[2])
        label_w2v = batch[3]
        
        self.sentences_w2v = pad_sequence(sentences_w2v, padding_value=0, batch_first=True)
        self.lengths = lengths
        self.label_onehot = label_onehot
        self.label_w2v = label_w2v

def batch_function(batch):
    return IntentBatch(batch)