import os
import time
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
device = torch.device("cpu")

from transformers import BertConfig, BertTokenizer, BertModel, BertForTokenClassification
cls_token='[CLS]'
eos_token='[SEP]'
unk_token='[UNK]'
pad_token='[PAD]'
mask_token='[MASK]'
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
TheModel = BertModel
ModelForTokenClassification = BertForTokenClassification

labels = ['B-assist', 'I-assist', 'B-cellno', 'I-cellno', 'B-city', 'I-city', 'B-community', 'I-community', 'B-country', 'I-country', 'B-devZone', 'I-devZone', 'B-district', 'I-district', 'B-floorno', 'I-floorno', 'B-houseno', 'I-houseno', 'B-otherinfo', 'I-otherinfo', 'B-person', 'I-person', 'B-poi', 'I-poi', 'B-prov', 'I-prov', 'B-redundant', 'I-redundant', 'B-road', 'I-road', 'B-roadno', 'I-roadno', 'B-roomno', 'I-roomno', 'B-subRoad', 'I-subRoad', 'B-subRoadno', 'I-subRoadno', 'B-subpoi', 'I-subpoi', 'B-subroad', 'I-subroad', 'B-subroadno', 'I-subroadno', 'B-town', 'I-town']
label2id = {}
for i, l in enumerate(labels):
    label2id[l] = i
num_labels = len(labels)

config = BertConfig.from_pretrained('bert-base-chinese')
class BertForSeqTagging(ModelForTokenClassification):
    def __init__(self):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = TheModel.from_pretrained('bert-base-chinese')
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.init_weights()
            
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        batch_size, max_len, feature_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return active_logits

model = BertForSeqTagging()
model.to(device)
model.load_state_dict(torch.load('Neural_Chinese_Address_Parsing_BERT_state_dict.pkl', map_location=torch.device('cpu')))
def get_address_info(address):
    # address = "北京市海淀区西土城路10号北京邮电大学"
    input_token = [cls_token] + list(address) + [eos_token]
    input_ids = tokenizer.convert_tokens_to_ids(input_token)
    attention_mask = [1] * (len(address) + 2)
    ids = torch.LongTensor([input_ids])
    atten_mask = torch.LongTensor([attention_mask])
    x = model(ids, atten_mask)
    logits = model(ids, atten_mask)
    logits = F.softmax(logits, dim=-1)
    logits = logits.data.cpu()
    rr = torch.argmax(logits, dim=1)
    import collections
    r = collections.defaultdict(list)
    for i, x in enumerate(rr.numpy().tolist()[1:-1]):
        r[labels[x][2:]].append(address[i])
    return r