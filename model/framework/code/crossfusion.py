import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd

class MultiHeadAttention(layers.Layer):
    def __init__(self, hidden_dim, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.all_head_dim = self.num_heads * self.head_dim
        
        self.query = layers.Dense(self.all_head_dim)
        self.key = layers.Dense(self.all_head_dim)
        self.value = layers.Dense(self.all_head_dim)
        
        self.dropout = layers.Dropout(dropout_rate)
        self.output_proj = layers.Dense(hidden_dim)
    
    def transpose_for_scores(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return tf.transpose(x, [0, 2, 1, 3])  # [batch, heads, seq_len, head_dim]
    
    def call(self, q, k, v, mask=None, training=None):
        q_proj = self.query(q)  # [batch, seq_len_q, all_head_dim]
        k_proj = self.key(k)    # [batch, seq_len_k, all_head_dim]
        v_proj = self.value(v)  # [batch, seq_len_v, all_head_dim]
        
        q_heads = self.transpose_for_scores(q_proj)  # [batch, heads, seq_len_q, head_dim]
        k_heads = self.transpose_for_scores(k_proj)  # [batch, heads, seq_len_k, head_dim]
        v_heads = self.transpose_for_scores(v_proj)  # [batch, heads, seq_len_v, head_dim]
        
        # Calculate attention score
        attention_scores = tf.matmul(q_heads, k_heads, transpose_b=True)  # [batch, heads, seq_len_q, seq_len_k]
        attention_scores = attention_scores / tf.math.sqrt(float(self.head_dim))
        
        # Apply mask (if have)
        if mask is not None:
            # mask shape: [batch, 1, seq_len_q, seq_len_k]
            mask = tf.expand_dims(mask, axis=1)
            adder = (1.0 - tf.cast(mask, tf.float32)) * -10000.0
            attention_scores = attention_scores + adder
        
        # Calculate attention weight
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)  # [batch, heads, seq_len_q, seq_len_k]
        attention_probs = self.dropout(attention_probs, training=training)
        
        # Apply attention
        context = tf.matmul(attention_probs, v_heads)  # [batch, heads, seq_len_q, head_dim]
        context = tf.transpose(context, [0, 2, 1, 3])  # [batch, seq_len_q, heads, head_dim]
        context = tf.reshape(context, [tf.shape(context)[0], tf.shape(context)[1], self.all_head_dim])
        
        output = self.output_proj(context)  # [batch, seq_len_q, hidden_dim]
        
        return output, attention_probs

class FeedForward(layers.Layer):
    def __init__(self, hidden_dim, intermediate_dim, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.dense1 = layers.Dense(intermediate_dim, activation="gelu")
        self.dense2 = layers.Dense(hidden_dim)
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, x, training=None):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return x

class CrossAttentionLayer(layers.Layer):
    def __init__(self, hidden_dim, num_heads, intermediate_dim, dropout_rate=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout_rate)
        self.attention_norm = layers.LayerNormalization(epsilon=1e-12)
        self.ffn = FeedForward(hidden_dim, intermediate_dim, dropout_rate)
        self.ffn_norm = layers.LayerNormalization(epsilon=1e-12)
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, q, k, v, mask=None, training=None):
        # self-attention
        attn_output, attention_probs = self.attention(q, k, v, mask, training=training)
        attn_output = self.dropout(attn_output, training=training)
        attn_output = self.attention_norm(q + attn_output) 
        
        # forward
        ffn_output = self.ffn(attn_output, training=training)
        ffn_output = self.ffn_norm(attn_output + ffn_output)  
        
        return ffn_output, attention_probs

class MolecularCrossAttention(keras.Model):
    """
    Five molecule embedding inputs:
    - sequence-type: ChemBERT, SMoleBERT, Molformer
    - graph-type: MoleBERT, MolCLR
    """
    def __init__(self, hidden_dim=768, num_heads=12, intermediate_dim=3072, num_layers=1, dropout_rate=0.1):
        super(MolecularCrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.seq_layers = [
            CrossAttentionLayer(hidden_dim, num_heads, intermediate_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.graph_layers = [
            CrossAttentionLayer(hidden_dim, num_heads, intermediate_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.seq_to_graph_layers = [
            CrossAttentionLayer(hidden_dim, num_heads, intermediate_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.graph_to_seq_layers = [
            CrossAttentionLayer(hidden_dim, num_heads, intermediate_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.seq_output = layers.Dense(hidden_dim)
        self.graph_output = layers.Dense(hidden_dim)
        
    def call(self, inputs, training=None):
        chembert_hidden, smolebert_hidden, molformer_hidden, molebert_hidden, molclr_hidden = inputs
        
        seq_embeddings = tf.concat([chembert_hidden, smolebert_hidden, molformer_hidden], axis=1)
        
        graph_embeddings = tf.concat([molebert_hidden, molclr_hidden], axis=1)
        
        seq_hidden = seq_embeddings
        for layer in self.seq_layers:
            seq_hidden, _ = layer(seq_hidden, seq_hidden, seq_hidden, training=training)
        
        graph_hidden = graph_embeddings
        for layer in self.graph_layers:
            graph_hidden, _ = layer(graph_hidden, graph_hidden, graph_hidden, training=training)
        
        seq_to_graph_hidden = seq_hidden
        for layer in self.seq_to_graph_layers:
            seq_to_graph_hidden, seq_to_graph_attn = layer(
                seq_hidden, graph_hidden, graph_hidden, training=training
            )
        
        graph_to_seq_hidden = graph_hidden
        for layer in self.graph_to_seq_layers:
            graph_to_seq_hidden, graph_to_seq_attn = layer(
                graph_hidden, seq_hidden, seq_hidden, training=training
            )
        
        seq_output = self.seq_output(seq_to_graph_hidden)
        graph_output = self.graph_output(graph_to_seq_hidden)
        
        return seq_output, graph_output, seq_to_graph_attn, graph_to_seq_attn

def create_molecular_model(seq_len=1, graph_len=1, feature_dim=768, hidden_dim=768, num_classes=1):
    """
    create a cross-attention module for molecules
    
    params:
    - seq_len
    - graph_len
    - feature_dim
    - hidden_dim
    - num_classes
    
    return:
    - model: Keras model
    """
    chembert_input = keras.Input(shape=(seq_len, feature_dim), name="chembert_input")
    smolebert_input = keras.Input(shape=(seq_len, feature_dim), name="smolebert_input")
    molformer_input = keras.Input(shape=(seq_len, feature_dim), name="molformer_input")
    molebert_input = keras.Input(shape=(graph_len, feature_dim), name="molebert_input")
    molclr_input = keras.Input(shape=(graph_len, feature_dim), name="molclr_input")
    
    cross_attention = MolecularCrossAttention(
        hidden_dim=hidden_dim,
        num_heads=12,
        intermediate_dim=hidden_dim*4,
        num_layers=1,
        dropout_rate=0.1
    )
    
    seq_output, graph_output, _, _ = cross_attention(
        [chembert_input, smolebert_input, molformer_input, molebert_input, molclr_input]
    )
    
    seq_pooled = layers.GlobalAveragePooling1D()(seq_output)
    graph_pooled = layers.GlobalAveragePooling1D()(graph_output)
    
    merged = layers.Concatenate()([seq_pooled, graph_pooled])
    
    feature_extractor = keras.Model(
        inputs=[chembert_input, smolebert_input, molformer_input, molebert_input, molclr_input],
        outputs=merged
    )
    
    return feature_extractor

def adjust_feature_dim(df, target_dim):
    current_dim = df.shape[1]
    # print(f"Current feature dimension: {current_dim}, target feature dimension: {target_dim}")
    
    # linear model to transform dimension
    model = keras.Sequential([
        keras.layers.Dense(target_dim, input_shape=(current_dim,))
    ])
    model.compile(optimizer='adam', loss='mse')
    
    data = df.values
    new_data = model.predict(data, verbose=0)
    
    print(f"Current feature dimension: {new_data.shape[1]}")
    return new_data

def load_and_preprocess_data():
    print("Load from existing files...")

    chembert_df = pd.read_csv('./data/bert/d2/chem2.csv')
    smolebert_df = pd.read_csv('./data/bert/d2/smole2.csv')
    molformer_df = pd.read_csv('./data/bert/d2/molformer2.csv')
    molebert_df = pd.read_csv('./data/bert/d2/mole2.csv')
    molclr_df = pd.read_csv('./data/bert/d2/molclr2.csv')
    y_df = pd.read_csv('./data/bert/d2/label2.csv')
    
    # 确认所有数据具有相同的样本数
    num_samples = chembert_df.shape[0]
    print(f"样本数量: {num_samples}")
    
    # 删除Activity列（保持为DataFrame）
    chembert_df = chembert_df.drop('Activity', axis=1)
    smolebert_df = smolebert_df.drop('Activity', axis=1)
    molformer_df = molformer_df.drop('Activity', axis=1)
    molebert_df = molebert_df.drop('Activity', axis=1)
    molclr_df = molclr_df.drop('Activity', axis=1)
    
    # 设置目标特征维度
    feature_dim = 768
    
    # 调整每个特征到相同的维度
    print("\n处理 ChemBERT 特征:")
    chembert_data = adjust_feature_dim(chembert_df, feature_dim)
    print(f"ChemBERT 数据形状: {chembert_data.shape}")
    print(f"ChemBERT 数据元素数量: {chembert_data.size}")
    
    print("\n处理 SMoleBERT 特征:")
    smolebert_data = adjust_feature_dim(smolebert_df, feature_dim)
    print(f"SMoleBERT 数据形状: {smolebert_data.shape}")
    print(f"SMoleBERT 数据元素数量: {smolebert_data.size}")
    
    print("\n处理 Molformer 特征:")
    molformer_data = adjust_feature_dim(molformer_df, feature_dim)
    print(f"Molformer 数据形状: {molformer_data.shape}")
    print(f"Molformer 数据元素数量: {molformer_data.size}")
    
    print("\n处理 MoleBERT 特征:")
    molebert_data = adjust_feature_dim(molebert_df, feature_dim)
    print(f"MoleBERT 数据形状: {molebert_data.shape}")
    print(f"MoleBERT 数据元素数量: {molebert_data.size}")
    
    print("\n处理 MolCLR 特征:")
    molclr_data = adjust_feature_dim(molclr_df, feature_dim)
    print(f"MolCLR 数据形状: {molclr_data.shape}")
    print(f"MolCLR 数据元素数量: {molclr_data.size}")
    
    # 调整输入维度为 (样本数, 1, 特征数)
    chembert_data = chembert_data.reshape(num_samples, 1, feature_dim)
    smolebert_data = smolebert_data.reshape(num_samples, 1, feature_dim)
    molformer_data = molformer_data.reshape(num_samples, 1, feature_dim)
    molebert_data = molebert_data.reshape(num_samples, 1, feature_dim)
    molclr_data = molclr_data.reshape(num_samples, 1, feature_dim)
    
    # 准备输入数据
    x_data = {
        "chembert_input": chembert_data,
        "smolebert_input": smolebert_data,
        "molformer_input": molformer_data,
        "molebert_input": molebert_data,
        "molclr_input": molclr_data
    }
    
    y_data = y_df.values
    
    # 验证最终数据形状
    print("\n最终数据形状:")
    for name, data in x_data.items():
        print(f"{name} shape: {data.shape}")
    print(f"标签形状: {y_data.shape}")
    
    return x_data, y_data

def create_prediction_model(input_dim):
    """create model based on merged feature"""
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # binary
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model
