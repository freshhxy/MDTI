import torch

from transformers import GPT2Tokenizer, GPT2Model
import os
import h5py
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from Data_process.data_process import load_dataset
from args import arguments
# 假设 pattern 是一个形状为 (16, 3, 1) 的张量，表示 16 个聚类模式
# 假设 input_sequence 是一个形状为 (64, 12, 170, 1) 的张量，表示输入序列

# 将每个样本的 12 个时间步划分为 4 个模式长度（每个模式 3 个时间步）
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained('./TimeCMA/gpt2')
model = GPT2Model.from_pretrained('./TimeCMA/gpt2').to(device)

def split_pattern(input_sequence):
    total_time_steps, channels = input_sequence.shape
    pattern_length = 3  # 每个模式的长度
    num_patterns = total_time_steps // pattern_length  # 每个样本的模式数量
    # 重塑输入序列形状为 (批次大小, 模式数量, 模式长度, 特征维度, 通道数)
    input_split = input_sequence.reshape(num_patterns, pattern_length, channels)
    # 提取每个模式的特征维度（这里简化为只取第一个特征的值，可以根据实际需求调整）
    input_split_features = input_split  # 取第一个特征，形状变为 (num_patterns, pattern_length, 1)
    return input_split_features

# 计算输入模式与每个聚类模式之间的相似性分数
def compute_similarity(input_split_features, pattern):
    num_patterns, pattern_length, _ = input_split_features.shape
    num_clusters = pattern.shape[0]
    # 初始化相似性分数矩阵
    similarity_scores = np.zeros((num_patterns, num_clusters))
    # 计算每个模式与每个聚类模式之间的欧几里得距离
    for i in range(num_patterns):
        input_pattern = input_split_features[i, :, :]  # 当前模式的特征，形状为 (pattern_length, 1)
        input_pattern_flat = input_pattern.reshape(1, -1)  # 展平为 (1, pattern_length * 1)
        pattern_flat = pattern.reshape(num_clusters, -1)  # 展平为 (num_clusters, pattern_length * 1)
        distances = euclidean_distances(input_pattern_flat, pattern_flat)
        # 将距离转换为相似性分数（例如，使用高斯核）
        similarities = np.exp(-distances / (2 * np.sqrt(pattern_length)))  # 使用高斯核
        similarity_scores[i, :] = similarities
    return similarity_scores


# 计算每个模式对应每个类别的概率
def compute_probabilities(similarity_scores):
    # 使用 softmax 函数将相似性分数转换为概率分布
    exponentials = np.exp(similarity_scores)
    probabilities = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    return probabilities

# 生成提示词
def _prepare_prompt(input,input_template,i,j,pattern_keys):
    index=input[i, :, j,0]
    input=input[...,1:].astype(np.float32)
    input=torch.from_numpy(input).float()
    values = input[i, :, j].flatten().tolist()

    values_str = ", ".join([str(value) for value in values])

    trends = torch.sum(torch.diff(input[i, :, j].flatten()))

    trends_str = f"{trends.item():0f}"

    for p in range(4):
        input_split_features=split_pattern(input[i, :, j])
        similarity_scores = compute_similarity(input_split_features, pattern_keys)
        probability = compute_probabilities(similarity_scores)
        prob = probability[p]
        top_categories = np.argsort(prob)[::-1][:3]
        top_probabilities = prob[top_categories]
        top_patterns = pattern_keys.squeeze(-1)[top_categories]



    top_probabilities_str = ", ".join([str(value) for value in top_probabilities.tolist()])
    top_categories_str = ", ".join([str(i) for i in top_categories.tolist()])
    top_patterns_str = ", ".join([str(i) for i in top_patterns.tolist()])

    in_prompt = input_template.replace("value1, ..., valuen", values_str)
    in_prompt = in_prompt.replace("Trends", trends_str)
    in_prompt = in_prompt.replace("pattern1, ..., patternn", top_patterns_str)
    in_prompt = in_prompt.replace("top_categories", top_categories_str)
    in_prompt = in_prompt.replace("top_probabilities", top_probabilities_str)

    in_prompt = in_prompt.replace("[t1]",str(index[0])).replace("[t2]", str(index[-1]))



    tokenized_prompt = tokenizer.encode(in_prompt, return_tensors="pt").to(device)


    return tokenized_prompt

def forward( tokenized_prompt):
        with torch.no_grad():
            prompt_embeddings = model(tokenized_prompt).last_hidden_state
        return prompt_embeddings


def generate_prompt(input_sequence,pattern_keys):

    input_templates = {
        'PEMS': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends. The patterns are sequentially pattern1, ..., patternn, with corresponding categories and probabilities of top_categories and top_probabilities respectively.",
        'ILI': "From [t1] to [t2], the values were value1, ..., valuen every week. The total trend value was Trends",
        'ETTh1': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
        'ETTh2': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
        'ECL': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
        'ETTm1': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
        'ETTm2': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
        'Weather': "From [t1] to [t2], the values were value1, ..., valuen every 10 minutes. The total trend value was Trends"
    }

    input_template = input_templates['PEMS']

    batch_size,input_len,num_node,dim = input_sequence.shape

    tokenized_prompts = []
    max_token_count=0

    for i in range(batch_size):
        for j in range(num_node):
            tokenized_prompt= _prepare_prompt(input_sequence, input_template, i, j,pattern_keys)
            max_token_count = max(max_token_count, tokenized_prompt.shape[1])

            tokenized_prompts.append((i, tokenized_prompt.to(device), j))

    in_prompt_emb = torch.zeros((len(input_sequence), max_token_count, 768, input_sequence.shape[2]), dtype=torch.float32,
                                device=device)

    last_token_emb=0

    for i, tokenized_prompt, j in tokenized_prompts:
        prompt_embeddings = forward(tokenized_prompt)
        padding_length = max_token_count - tokenized_prompt.shape[1]
        if padding_length > 0:
            last_token_embedding = prompt_embeddings[:, -1, :].unsqueeze(1)
            padding = last_token_embedding.repeat(1, padding_length, 1)
            prompt_embeddings_padded = torch.cat([prompt_embeddings, padding], dim=1)
        else:
            prompt_embeddings_padded = prompt_embeddings


        in_prompt_emb[i, :max_token_count, :, j] = prompt_embeddings_padded
        last_token_emb = in_prompt_emb[:, max_token_count - 1:max_token_count, :, :]


    return last_token_emb



if __name__ == '__main__':
    args = arguments()
    data_set ,pattern_keys= load_dataset(args.dataset_dir, args.dataset_name, args.seq_len, args.traffic_state, args.batch_size,'Road')

    """


    data_train=data_set['train_loader'].get_iterator ()

    device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    save_path = f"./Embeddings/{args.dataset_name}/{args.traffic_state}/"
    os.makedirs(save_path, exist_ok=True)

    emb_time_path = f"./Results/emb_logs/"
    os.makedirs(emb_time_path, exist_ok=True)


    for i, (x,y)in enumerate(data_train):
        print('x,shape',x.shape)

        embeddings = generate_prompt(x, pattern_keys)

        print(embeddings)

        file_path = f"{save_path}{'train'}.h5"
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('embeddings', data=embeddings.cpu().numpy())
            
    
    

    data_train = data_set['test_loader'].get_iterator()

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    save_path = f"./Embeddings/{args.dataset_name}/{args.traffic_state}/"
    os.makedirs(save_path, exist_ok=True)

    emb_time_path = f"./Results/emb_logs/"
    os.makedirs(emb_time_path, exist_ok=True)

    for i, (x, y) in enumerate(data_train):
        print('x,shape', x.shape)

        embeddings = generate_prompt(x, pattern_keys)

        print(embeddings)

        file_path = f"{save_path}{'test'}.h5"
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('embeddings', data=embeddings.cpu().numpy())
            
            
    

    data_train = data_set['val_loader'].get_iterator()

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    save_path = f"./Embeddings/{args.dataset_name}/{args.traffic_state}/"
    os.makedirs(save_path, exist_ok=True)

    emb_time_path = f"./Results/emb_logs/"
    os.makedirs(emb_time_path, exist_ok=True)

    for i, (x, y) in enumerate(data_train):
        print('x,shape', x.shape)

        embeddings = generate_prompt(x, pattern_keys)

        print(embeddings)

        file_path = f"{save_path}{'val'}.h5"
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('embeddings', data=embeddings.cpu().numpy())
            
    """





