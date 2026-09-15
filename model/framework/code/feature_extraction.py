import pandas as pd
import numpy as np
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from transformers import BertModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')
import random
from transformers import set_seed
from Capsule_MPNN import *
from Mole_Bert import molebert
from ginet_molclr import molclr
from MolFormer import molformer
from frozen_fusion import load_frozen_fusion

# ERSILIA FIX: upstream set os.environ["HF_HOME"] to a URL, but HF_HOME must
# be a local cache directory - as written this breaks transformers' cache
# resolution. Left unset so the default local cache is used; see
# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
# for HF_ENDPOINT if a mirror is genuinely needed in a specific environment.

FUSION_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints", "fusion_weights.weights.h5"
)


def smolebert(smiles_list):
    tokenizer = AutoTokenizer.from_pretrained("UdS-LSV/smole-bert")
    model = BertModel.from_pretrained("UdS-LSV/smole-bert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = len(smiles_list)
    list2 = []
    for i in range(x):
        print(i)
        sequence_Example = smiles_list[i]
        print('SMILE-length:', len(sequence_Example))
        max_length = model.config.max_position_embeddings
        encoded_input = tokenizer(sequence_Example, max_length=max_length, return_tensors='pt')

        output = model(**encoded_input)
        s = output[0].data.cpu().numpy()
        list2.append(s)

    list1 = []
    for i in range(x):
        data = list2[i]
        d = data.mean(axis=1)
        feat = d[0].tolist()
        list1.append(feat)

    feature_df = pd.DataFrame(list1)
    return feature_df


def chemmolefusion(smiles_list):
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    set_seed(seed_value)
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    model = BertModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

    x = len(smiles_list)
    list2 = []
    for i in range(x):
        print(i)
        sequence_Example = smiles_list[i]
        print('SMILE-length:', len(sequence_Example))
        max_length = model.config.max_position_embeddings
        encoded_input = tokenizer(sequence_Example, max_length=max_length, return_tensors='pt')

        output = model(**encoded_input)
        s = output[0].data.cpu().numpy()
        list2.append(s)
    list1 = []
    for i in range(x):
        data = list2[i]
        d = data.mean(axis=1)
        feat = d[0].tolist()
        list1.append(feat)

    feature_chembert = pd.DataFrame(list1)
    feature_mole = molebert(smiles_list)
    feature_smole = smolebert(smiles_list)
    feature_molclr = molclr(smiles_list)
    feature_molformer = molformer(smiles_list)
    # feature_fusion = pd.concat([feature_chembert, feature_mole, feature_smole, feature_molclr], axis=1) ###

    return feature_chembert, feature_mole, feature_smole, feature_molclr, feature_molformer

from tensorflow.keras import layers
from crossfusion import MultiHeadAttention,FeedForward,CrossAttentionLayer,MolecularCrossAttention,create_molecular_model

def cross_fusion(smiles_list):
    chembert_feat, molebert_feat, smolebert_feat, molclr_feat, molformer_feat = chemmolefusion(smiles_list)
    
    feature_dim = 768
    embeddings = {
        "chembert": chembert_feat.values if hasattr(chembert_feat, 'values') else chembert_feat,
        "molebert": molebert_feat.values if hasattr(molebert_feat, 'values') else molebert_feat,
        "smolebert": smolebert_feat.values if hasattr(smolebert_feat, 'values') else smolebert_feat,
        "molclr": molclr_feat.values if hasattr(molclr_feat, 'values') else molclr_feat,
        "molformer": molformer_feat.values if hasattr(molformer_feat, 'values') else molformer_feat
    }
    
    processed_data = {}
    for name, data in embeddings.items():
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        if data.shape[1] != feature_dim:
            print(f"Adjust {name}-feature dimension: {data.shape[1]} -> {feature_dim}")
            input_dim = data.shape[1]
            
            adjust_model = tf.keras.Sequential([
                layers.Dense(feature_dim, input_shape=(input_dim,), activation='linear')
            ])
            adjust_model.compile(optimizer='adam', loss='mse')
            
            data = adjust_model.predict(data, verbose=0)
        
        processed_data[f"{name}_input"] = np.expand_dims(data, axis=1)
    
    return processed_data

def get_feature(input_path, drug_descripter):
    """
    提取分子特征
    参数:
        input_path: 输入文件路径
        drug_descriptor: 特征类型 ['fusion', 'chembert', 'molebert', 'smolebert', 'molclr', 'molformer']
        output_dir: 输出目录
    返回:
        DataFrame: 包含提取的特征
    """
    input_df = pd.read_csv(input_path)

    if 'SMILES' not in input_df.columns:
        raise ValueError("Lack of 'SMILES' column input.")
        
    smiles_list = input_df['SMILES'].tolist()
    has_activity = 'Activity' in input_df.columns
    activity_col = input_df['Activity'] if has_activity else None

    output_dir='./data'
    os.makedirs(output_dir, exist_ok=True)
    
    if drug_descripter == 'fusion':
        try:
            fusion_input = cross_fusion(smiles_list)
            # ERSILIA FIX: the upstream create_molecular_model() is untrained
            # and re-randomized on every call, making features (and therefore
            # predictions) non-reproducible across process runs. Load the
            # frozen weights built once via frozen_fusion.py instead - see
            # that file's docstring for details.
            fusion_model = load_frozen_fusion(FUSION_WEIGHTS_PATH)
            features = fusion_model.predict(fusion_input)
            
            output_df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
            
            if has_activity:
                output_df['Activity'] = activity_col.values
                
            # output_path = os.path.join(output_dir, 'fused_features.csv')
            # output_df.to_csv(output_path, index=False)
            return output_df
            
        except Exception as e:
            print(f"Failed to get feature - {str(e)}")
            exit(1)
    else:
        raise ValueError(f"Not support: {drug_descripter}")
    
if __name__ == "__main__":
    import argparse
    import os
    import numpy as np
    import pandas as pd
    import tensorflow as tf  # 添加TensorFlow导入
    
    parser = argparse.ArgumentParser(description='Molecular feature extraction and fusion.')
    parser.add_argument('--feature_type', type=str, required=True,
                       choices=['fusion', 'chembert', 'molebert', 'smolebert', 'molclr', 'molformer', 'concat'], 
                       help='feature_type')
    parser.add_argument('--input_path', type=str, required=True,
                       help='input directory')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        input_df = pd.read_csv(args.input_path)
        if 'SMILES' not in input_df.columns:
            raise ValueError("Lack of 'SMILE' column input.")
        
        smiles_list = input_df['SMILES'].tolist()
        has_activity = 'Activity' in input_df.columns
        activity_col = input_df['Activity'] if has_activity else None
        print('Get SMILES list.')
        
    except Exception as e:
        print(f"Failed: can't load - {str(e)}")
        exit(1) 

    if args.feature_type == 'fusion':
        try:
            fusion_input = cross_fusion(smiles_list)
            fusion_model = create_molecular_model()
            features = fusion_model.predict(fusion_input)
            
            output_df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
            
            if has_activity:
                output_df['Activity'] = activity_col.values
                
            output_path = os.path.join(args.output_dir, 'fused_features.csv')
            output_df.to_csv(output_path, index=False)
            
        except Exception as e:
            print(f"Failed to get feature - {str(e)}")
            exit(1)
    else:
        try:
            chembert_feat, molebert_feat, smolebert_feat, molclr_feat, molformer_feat = chemmolefusion(smiles_list)
            
            if args.feature_type == 'chembert':
                features = chembert_feat
            elif args.feature_type == 'molebert':
                features = molebert_feat
            elif args.feature_type == 'smolebert':
                features = smolebert_feat
            elif args.feature_type == 'molclr':
                features = molclr_feat
            elif args.feature_type == 'molformer':
                features = molformer_feat

            output_df = pd.DataFrame(features.values)
            
            if has_activity:
                output_df['Activity'] = activity_col.values
            
            output_path = os.path.join(args.output_dir, f'{args.feature_type}_features.csv')
            output_df.to_csv(output_path, index=False)
            

        except Exception as e:
            print(f"Failed to get feature: {args.feature_type} - {str(e)}")
            exit(1)


    print(f"Get feature in: {output_path}")
