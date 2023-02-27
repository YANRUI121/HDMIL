import os
import glob
import random
import json


train_dir = r'/home/sdd/zxy/TCGA_data/all_mil_patch/FGFR3/train'
valid_dir = r'/home/sdd/zxy/TCGA_data/all_mil_patch/FGFR3/valid'
test_dir = r'/home/sdd/zxy/TCGA_data/all_mil_patch/FGFR3/test'
txt_dir = r'/home/sdd/zxy/TCGA_data/all_mil_patch/FGFR3'
mutation_type = '1_mutation'
wild_type = '0_wild'
num_count_mutation = 36 
num_count_wild = 6 


def write_txt(file_list, txt_path):
    with open(txt_path, 'w') as f:
        json.dump(file_list, f)


def make_list(file_dir, gene_type):
    file_list = glob.glob(file_dir + '/*_0')
    all_list = []
    for file in file_list:
        file_num_list = []
        if gene_type == 'mutation':
            num_count=100
            num_count_type = num_count_mutation
        else:
            num_count=50
            num_count_type = num_count_wild
        num_list = random.sample(range(0, num_count), num_count_type)
        for num in num_list:
            file_num_list.append(file.replace('_0', '_{}'.format(str(num))))
        all_list.append(file_num_list)
    return all_list


def run():
    mutation_train_dir = os.path.join(train_dir, mutation_type)
    wild_train_dir = os.path.join(train_dir, wild_type)
    mutation_valid_dir = os.path.join(valid_dir, mutation_type)
    wild_valid_dir = os.path.join(valid_dir, wild_type)
    mutation_test_dir = os.path.join(test_dir, mutation_type)
    wild_test_dir = os.path.join(test_dir, wild_type)
    train_list = make_list(mutation_train_dir, 'mutation') + make_list(wild_train_dir, 'wild')
    valid_list = make_list(mutation_valid_dir, 'mutation') + make_list(wild_valid_dir, 'wild')
    test_list = make_list(mutation_test_dir, 'mutation') + make_list(wild_test_dir, 'wild')
    print('train length:', len(train_list), '   valid length:', len(valid_list), '  test length:', len(test_list))
    write_txt(train_list, os.path.join(txt_dir, 'train_m{}_w{}.json'.format(str(num_count_mutation), str(num_count_wild))))
    write_txt(valid_list, os.path.join(txt_dir, 'valid_m{}_w{}.json'.format(str(num_count_mutation), str(num_count_wild))))
    write_txt(test_list, os.path.join(txt_dir, 'test_m{}_w{}.json'.format(str(num_count_mutation), str(num_count_wild))))



if __name__ == '__main__':
    run()
