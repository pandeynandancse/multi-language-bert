import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        #load multilingual bert model
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
	
	    #bert base uncased has 768 outputs 
        # 1 becoz binary calssification problem
        #also multiply by 2 becoz mean and max pooling has been applied 
        self.out = nn.Linear(768*2, 1)

    def forward(self, ids, mask, token_type_ids):
        #here output o1 has been used ==>>> if u  want then you can use output o2 as shown in sentiment-with-bert-onnx repository
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

	    #a lot of things can be done here for best model but let keep it simple ===>> see bert documentation
        
        mean_pooling = torch.mean(o1, 1)
        max_pooling, _ = torch.max(o1,1)
        cat = torch.cat{ (mean_pooling,max_pooling,1) }

        bo = self.bert_drop(cat)
        output = self.out(bo)
        return output