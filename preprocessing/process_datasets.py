import sys
sys.path.append(".")

import pandas as pd
import json
import os
from CONSTANTS import DATA_DIR

def extract_csv(dataset):
    file_path = os.path.join(DATA_DIR[dataset], 'entity_idx.json')
    e2idx = json.load(open(file_path))

    file_path = os.path.join(DATA_DIR[dataset], 'rel_idx.json')
    rel2idx = json.load(open(file_path))

    splits = ['train', 'valid', 'test']
    for split in splits:
        data = []
        with open(os.path.join(DATA_DIR[dataset], f'{split}.txt')) as f:
            for i, line in enumerate(f):
                if 'FB15K' in dataset or 'WN18' in dataset:
                    e1, rel, e2 = line.split('\t')
                else:
                    rel, e1, e2 = line.split('\t')
                e1 = e1.strip()
                e2 = e2.strip()
                rel = rel.strip()
                rel_reverse = rel+ '_reverse'
                data.append([e1, e2idx[e1], rel, rel2idx[rel], e2, e2idx[e2]])
                data.append([e2, e2idx[e2], rel_reverse, rel2idx[rel_reverse], e1, e2idx[e1]])
        df = pd.DataFrame(data, columns = ['entity1', 'entity1_id', 'rel', 'rel_id', 'entity2', 'entity2_id'])
        # Observed some duplicated examples for CN82K
        # Maintain duplicates (if any) in eval splits for fair comparisons with other work
        if split == 'train':
            df.drop_duplicates(subset=['entity1_id', 'rel_id', 'entity2_id'], inplace=True)
        file_path = os.path.join(DATA_DIR[dataset], f'df_{split}.csv')
        print(f'Saving {file_path} with {len(df)//2} facts')
        df.to_csv(file_path)


def main():
    for dataset in ['WN18RR', 'FB15K_237', 'CN82K']:
        print(f'Processing {dataset}...')
        extract_csv(dataset)
    print('done')

if __name__ == '__main__':
    main()