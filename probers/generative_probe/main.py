import torch
import re
import argparse

def parse_result(model_name, front_text, gen_text):
    if model_name.startswith('facebook/bart-'):
        res = re.sub(front_text, '', gen_text)
        res = ' '.join(res.split()).strip()
        return res
    elif model_name.startswith('t5'):
        return gen_text.strip()
    elif model_name.startswith('razent/SciFive'):
        '''
        ['razent/SciFive-base-Pubmed_PMC', 'razent/SciFive-large-Pubmed_PMC', 
        'razent/SciFive-base-Pubmed', 'razent/SciFive-large-Pubmed',
        'SciFive-base-PMC', 'razent/SciFive-large-PMC']
        '''
        return gen_text.strip()
    else:
        raise Exception('Wrong model name!!!')
    
def initialize_model(model_name):
    if model_name.startswith('t5'):
        #print ('Use T5 Model')
        from transformers import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(model_name)

    elif model_name.startswith('razent/SciFive'):
        #print ('Use BioMedical T5')
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    elif model_name.startswith('facebook/bart-'):
        #print ('Use Bart Model')
        from transformers import BartForConditionalGeneration
        model = BartForConditionalGeneration.from_pretrained(model_name)
    else:
        raise Exception('Wrong Model Configuration!')  
    return model

def generative_probing(data, model, model_name, cuda_available, device):
    result_list, reference_list = [], []
    data_num = len(data.token_id_list)
    #data_num = 2
    import progressbar
    p = progressbar.ProgressBar(data_num)
    p.start()
    for idx in range(data_num):
        p.update(idx)
        token_id_tensor = data.token_id_list[idx]
        if cuda_available:
            token_id_tensor = token_id_tensor.cuda(device)
        front_text = data.front_text_list[idx]
        one_reference_list = data.reference_list[idx]
        _, src_len = token_id_tensor.size()
        outputs = model.generate(token_id_tensor, num_beams=10, num_return_sequences=10, max_length=src_len+80)
        one_res_list = []
        for item in outputs:
            gen_text = data.tokenized_decode(item)
            try:
                one_res_list.append(parse_result(model_name, front_text, gen_text))
            except:
                one_res_list.append(gen_text)
        result_list.append(one_res_list)
        reference_list.append(one_reference_list)
    p.finish()
    return result_list, reference_list

def process_one_path(prompt_path, data_path, model, model_name, result_prefix_path, cuda_available, device):
    from dataclass import Data
    data = Data(prompt_path, data_path, model_name)
    file_header_name = data_path.split(r'/')[-1]
    print ('Use {} model probe {} data'.format(model_name, file_header_name))
    probing_result_list, reference_result_list = \
    generative_probing(data, model, model_name, cuda_available, device)
    assert len(probing_result_list) == len(reference_result_list)

    save_list = []
    for idx in range(len(probing_result_list)):
        prob_list = probing_result_list[idx]
        ref_list = reference_result_list[idx]
        one_dict = {'reference':ref_list, model_name:prob_list}
        save_list.append(one_dict)

    save_path = result_prefix_path + '/' + re.sub('/','_',model_name) + '/'
    import os
    if os.path.exists(save_path):
        pass
    else: # recursively construct directory
        os.makedirs(save_path, exist_ok=True)
    save_name = save_path + file_header_name.strip(r'.csv').strip() + '_result.json'
    import json
    with open(save_name, 'w') as outfile:
        json.dump(save_list, outfile, indent=4)

def model_probe_all_path(data_prefix, prompt_path, model, model_name, result_prefix_path, cuda_available, device):
    import os
    file_name_list = os.listdir(data_prefix)
    for name in file_name_list:
        #if name.endswith(r'1000.csv'):
        if name.endswith(r'.csv'):
            data_path = data_prefix + '/' + name
            process_one_path(prompt_path, data_path, model, model_name, result_prefix_path, cuda_available, device)
        else:
            continue

def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--data_path_prefix', type=str, help='The path where the data stores.')
    parser.add_argument('--result_prefix_path', type=str, help='Where to save the result.')
    parser.add_argument('--prompt_path', type=str, help='Where to find the predefined prompt.')
    return parser.parse_args()


if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    else:
        print ('Cuda is not available.')
    cuda_available = torch.cuda.is_available()

    args = parse_config()
    device = torch.device('cuda')

    data_path_prefix = args.data_path_prefix
    result_prefix_path = args.result_prefix_path
    prompt_path = args.prompt_path

    #model_name_list = ['t5-large', 'facebook/bart-large', 'razent/SciFive-large-Pubmed_PMC', 'razent/SciFive-large-Pubmed',\
    #'razent/SciFive-large-PMC', 't5-base', 'facebook/bart-base', 'razent/SciFive-base-Pubmed_PMC', 'razent/SciFive-base-Pubmed',\
    #'SciFive-base-PMC', 't5-small']
    #model_name_list = ['facebook/bart-large', 'razent/SciFive-large-Pubmed_PMC',\
    #'razent/SciFive-large-PMC', 't5-base', 'facebook/bart-base', 'razent/SciFive-base-Pubmed_PMC', 'razent/SciFive-base-Pubmed',\
    #'SciFive-base-PMC', 't5-small']
    #model_name_list = ['facebook/bart-base', 't5-small', 'razent/SciFive-base-Pubmed_PMC', \
    #'razent/SciFive-base-Pubmed', 'razent/SciFive-base-PMC']
    #model_name_list = ['facebook/bart-base', 't5-small', 'razent/SciFive-base-Pubmed_PMC']
    model_name_list = ['t5-small', 'facebook/bart-base', 'razent/SciFive-base-Pubmed_PMC']

    for model_name in model_name_list:
        print ('-------------------  Use {} Model ---------------------'.format(model_name))
        model = initialize_model(model_name)
        if cuda_available:
            model = model.cuda(device)
        model.eval()
        model_probe_all_path(data_path_prefix, prompt_path, model, model_name, 
            result_prefix_path, cuda_available, device)

