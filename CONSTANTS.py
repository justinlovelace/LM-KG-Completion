SNOMED_CORE_DIR = 'data/UMLS/SNOMEDCT_CORE_SUBSET_202008'
DATA_DIR = {'SNOMED_CT_CORE': 'data/SNOMED-CT-Core/',
            'FB15K_237': 'data/FB15K-237/',
            'CN82K': 'data/CN82K/',
            'WN18RR': 'data/WN18RR/'}
COLUMN_NAMES = {'SNOMED_CT_CORE': ('CUI2_id', 'RELA_id', 'CUI1_id'),
                'FB15K_237': ('entity1_id', 'rel_id', 'entity2_id'),
                'CN82K': ('entity1_id', 'rel_id', 'entity2_id'),
                'WN18RR': ('entity1_id', 'rel_id', 'entity2_id')}
UMLS_SOURCE_DIR = 'data/UMLS/2020AA/META'
