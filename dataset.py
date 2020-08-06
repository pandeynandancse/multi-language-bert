import config
import torch


class BERTDataset:
    def __init__(self, common_text, target):
        self.common_text = common_text
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.common_text)

    def __getitem__(self, item):
	#get common_text
        common_text = str(self.common_text[item])
       
        #split common_text string into tokens
        common_text = " ".join(common_text.split())
	#encode plus can encode two string at a time -- but here we have only one --->> first is 'reivew' and second is 'None'

	#specials token are [CLS],[SEP] ==>>> eg. [cls] first_text [sep] second_text [sep] ===>> but here it will be  ==>> [cls]common_text[sep]
        inputs = self.tokenizer.encode_plus(
            common_text,

            #second string is none
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
	
	#token_type_ids in this case is going to be 1 for everything so we don't probably need it  becoz mask and token_type_ids are same in this case because string is only one that is 'review' only.
        token_type_ids = inputs["token_type_ids"]
	
	
	#perform padddig
	padding_length = self.max_len - len(ids)
	ids = ids + ([0] * padding_length)
	mask = mask + ([0] * padding_length)
	token_type_ids = token_type_ids + ([0] * padding_length)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
	    
            #torch.float because binary class , it depends on which loss function you are using
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }
