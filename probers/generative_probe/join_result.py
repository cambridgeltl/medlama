import os
import json

def get_all_json_name(prefix_path):
    folder_name_list = os.listdir(prefix_path)
    json_name_dict = {}
    for name in folder_name_list:
        one_path = prefix_path + '/' + name
        sub_name_list = os.listdir(one_path)
        for sub_name in sub_name_list:
            #header_name = sub_name.split('/')[-1].strip(r'_1000_result.json').strip()
            header_name = sub_name
            json_name_dict[header_name] = {}
    print (json_name_dict)
    print (len(json_name_dict))
    return json_name_dict

def load_one_result(path):
    with open(path) as f:
        result = json.load(f)
    all_keys_list = list(result[0].keys())
    res_dict = {}
    for key in all_keys_list:
        res_dict[key] = []
    for item in result:
        for key in res_dict:
            res_dict[key].append(item[key])
    return res_dict

def reshape_relation_dict(in_dict):
    key_list = list(in_dict.keys())
    case_num = len(in_dict[key_list[0]])
    res_list = []
    for idx in range(case_num):
        one_res_dict = {}
        for key in key_list:
            one_res_dict[key] = in_dict[key][idx]
        res_list.append(one_res_dict)
    return res_list

if __name__ == '__main__':
    prefix_path = r'./rel_probing_result/'
    json_name_dict = get_all_json_name(prefix_path)
    all_relation_list = list(json_name_dict.keys())
    sub_prefix_header_name_list = os.listdir(prefix_path)
    sub_prefix_list = [prefix_path + '/' + item + '/' for item in sub_prefix_header_name_list]
    tmp_res_dict = {}
    res_dict = {}
    for relation in all_relation_list:
        #one_relation_name = '_'.join(relation.split('_')[:-2]).strip('_')
        one_relation_name = '_'.join(relation.split('_')[:-1]).strip('_')
        tmp_res_dict[one_relation_name] = {}
        for one_sub_prefix in sub_prefix_list:
            one_path = one_sub_prefix + '/' + relation
            one_res_dict = load_one_result(one_path)
            for key in one_res_dict:
                if key in tmp_res_dict[one_relation_name]:
                    continue
                else:
                    tmp_res_dict[one_relation_name][key] = one_res_dict[key]
        one_relation_list = reshape_relation_dict(tmp_res_dict[one_relation_name])
        res_dict[one_relation_name] =  one_relation_list
        #print (len(res_dict[one_relation_name]))
        #print (res_dict[one_relation_name][0])
        #print (res_dict[one_relation_name][0].keys())
        #raise Exception()

    save_name = './biolama_all_char_len_result.json'
    with open(save_name, 'w') as outfile:
        json.dump(res_dict, outfile, indent=4)
