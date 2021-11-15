import re
import progressbar
import csv

class Data:
    def __init__(self, prompt_path, data_path, model_name):
        self.prompt_dict = self.load_prompt_dict(prompt_path)
        if model_name.startswith('t5'):
            #print ('Use T5 Model')
            from transformers import T5Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.placeholder_token = '<extra_id_0>'

        elif model_name.startswith('razent/SciFive'):
            #print ('Use BioMedical T5')
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.placeholder_token = '<extra_id_0>'

        elif model_name.startswith('facebook/bart-'):
            #print ('Use Bart Model')
            from transformers import BartTokenizer
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.placeholder_token = '<mask>'
        else:
            raise Exception('Wrong Model Configuration!')  

        self.special_token_list = ['<s>', '</s>', '<pad>', '.', '<extra_id_0>']
        self.data_list, self.token_id_list, self.front_text_list, self.reference_list = self.load_data(data_path)

    def load_prompt_dict(self, in_f):
        prompt_dict = {}
        with open(in_f, 'r', encoding = 'utf8') as i:
            lines = i.readlines()[1:]
            for l in lines:
                item_list = l.strip('\n').split(',')
                assert len(item_list) == 3
                relation = item_list[0].strip()
                human_prompt = item_list[1].strip('.').strip()
                prompt_dict[relation] = human_prompt
        return prompt_dict

    def process_one_instance(self, head_name, relation):
        prompt = self.prompt_dict[relation]
        prompt = re.sub(r'\[X\]', ' ' + head_name + ' ', prompt)
        prompt = ' '.join(prompt.split()).strip()
        front_text = re.sub(r'\[Y\]', '', prompt).strip()
        prompt = re.sub(r'\[Y\]', ' ' + self.placeholder_token + ' ', prompt)
        result = ' '.join(prompt.split()).strip()
        return result, front_text

    def load_data(self, in_f):
        with open(in_f) as f:
            data_num =  sum(1 for line in f)

        res_list, token_id_list, front_text_list, reference_list = [], [], [], []
        with open(in_f, 'r') as file:
            reader = csv.reader(file)
            p = progressbar.ProgressBar(data_num)
            p.start()
            idx = 0
            for item_list in reader:
                p.update(idx)
                if idx == 0: 
                    idx += 1
                    continue
                else:
                    idx += 1
                '''
                assert len(item_list) == 8
                head_name = item_list[1].strip()
                relation = item_list[2].strip()
                one_reference_list = item_list[3].split('||')
                '''
                head_name = item_list[2].strip()
                relation = item_list[3].strip()
                one_reference_list = item_list[4].split('||')
                one_reference_list = [one_ref.strip() for one_ref in one_reference_list]
                reference_list.append(one_reference_list)
                relation_list = relation.split('_')
                relation = ' '.join(relation_list).strip()
                one_res = [head_name, relation]
                one_text, one_front_text = self.process_one_instance(head_name, relation)
                one_input_id_tensor = self.tokenizer(one_text, return_tensors="pt").input_ids
                res_list.append(one_text)
                token_id_list.append(one_input_id_tensor)
                front_text_list.append(one_front_text)
            p.finish()
            print ('Data loaded.')
        return res_list, token_id_list, front_text_list, reference_list

    def tokenized_decode(self, token_id_list):
        pred_tokens = self.tokenizer.convert_ids_to_tokens(token_id_list)
        res_text = ''
        curr_list = []
        for token in pred_tokens:
            if token in self.special_token_list + ['<extra_id_0>', '<mask>']:
                if len(curr_list) == 0:
                    res_text += ' ' + token + ' '
                else:
                    curr_res = self.tokenizer.convert_tokens_to_string(curr_list)
                    res_text = res_text + ' ' + curr_res + ' ' + token + ' '
                    curr_list = []
            else:
                curr_list.append(token)

        if len(curr_list) > 0:
            curr_res = self.tokenizer.convert_tokens_to_string(curr_list)
            res_text = res_text + ' ' + curr_res + ' '
        res_text_list = res_text.strip().split()
        res_text = ' '.join(res_text_list).strip()
        res_list = res_text.split()
        final_res_list = []
        for token in res_list:
            if token in self.special_token_list:
                pass
            else:
                final_res_list.append(token)
        return ' '.join(final_res_list).strip()

