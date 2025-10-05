import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


# from config.config import Config # 假设 Config 可用

class GATBlockWithPrompt(nn.Module):
    # 假设 Config 包含 gpt2_max_len 等参数
    # 如果 Config 不可用，需要将这些参数作为 init 参数传入
    _input_templates = {
        'PEMS': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends. The patterns are sequentially pattern1, ..., patternn, with corresponding categories and probabilities of top_categories and top_probabilities respectively.",
        'ILI': "From [t1] to [t2], the values were value1, ..., valuen every week. The total trend value was Trends",
        'ETTh1': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
        'ETTh2': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
        'ECL': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
        'ETTm1': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
        'ETTm2': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
        'Weather': "From [t1] to [t2], the values were value1, ..., valuen every 10 minutes. The total trend value was Trends",
    }

    def __init__(self, dim_in, dim_out, heads, pattern_keys, gpt2_path='./TimeCMA/gpt2', device='cuda:0',
                 gpt2_max_len=1024):
        super().__init__()

        self.pattern_keys = pattern_keys  # 包含了模式数据和相关信息 (例如元组 (pattern_dict, array_with_other_info))
        self.device = device
        self.gpt2_max_len = gpt2_max_len  # 从Config或参数传入

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
        self.gpt2_model = GPT2Model.from_pretrained(gpt2_path).to(device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.attn = nn.MultiheadAttention(embed_dim=dim_in + 768, num_heads=heads, batch_first=True)
        self.fc = nn.Linear(dim_in + 768, dim_out)

    # --- 辅助函数：移入类内部 ---
    def _split_pattern(self, input_sequence):
        total_time_steps, channels = input_sequence.shape
        pattern_length = 3
        num_patterns = total_time_steps // pattern_length
        trimmed_length = num_patterns * pattern_length
        input_sequence = input_sequence[:trimmed_length]
        input_split_features = input_sequence.reshape(num_patterns, pattern_length, channels)
        return input_split_features

    def _compute_similarity(self, input_split_features, pattern_array_for_sim):
        # 这里的 pattern_array_for_sim 是从 self.pattern_keys 中提取的数值模式数组
        # 确保它已经是纯数值类型，没有字符串化的列表
        num_patterns, pattern_length, _ = input_split_features.shape
        num_clusters = pattern_array_for_sim.shape[0]
        similarity_scores = np.zeros((num_patterns, num_clusters))

        for i in range(num_patterns):
            input_pattern = input_split_features[i, :, :]
            input_pattern_flat = input_pattern.reshape(1, -1)
            pattern_flat = pattern_array_for_sim.reshape(num_clusters, -1)
            distances = euclidean_distances(input_pattern_flat, pattern_flat)
            similarities = np.exp(-distances / (2 * np.sqrt(pattern_length)))
            similarity_scores[i, :] = similarities
        return similarity_scores

    def _compute_probabilities(self, similarity_scores):
        exponentials = np.exp(similarity_scores)
        probabilities = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
        return probabilities

    def _prepare_single_prompt_tokenized(self, data_slice, node_idx, pattern_keys_tuple):
        # data_slice: input_sequence[i, :, j] (seq_len, dim) for a single batch item and node
        # node_idx: 当前节点的索引
        # pattern_keys_tuple: 整个 self.pattern_keys 元组

        # 从 pattern_keys_tuple 中获取实际用于相似度计算的模式数组
        # 假设 pattern_keys_tuple[0] 是一个字典，其中包含 'x_train' 作为模式数值数组
        # 同时，如果 pattern_keys_tuple[1] 包含了模式的非数值描述（如ID、文本），可以在这里使用

        # 再次强调：这里假设 pattern_keys_tuple[0]['x_train'] 已经是纯数值 np.ndarray
        pattern_numerical_array = pattern_keys_tuple[0]['x_train']

        # 您的原始代码中 `top_patterns` 是从 `pattern_keys.squeeze(-1)` 获取的
        # 这意味着 `pattern_keys` 的某个元素是需要被 squeeze 的。
        # 这里我假设 `pattern_keys_tuple[1]` 是另一个 NumPy 数组，包含了可以用于 `top_patterns` 的数据
        # 比如：pattern_keys_tuple[1] = np.array([[id1, id2, ...], ...])
        # 如果不是，请根据实际结构调整
        pattern_descriptive_array = pattern_keys_tuple[1]

        time_indices = data_slice[:, 0].long().tolist()  # 假设时间索引在第一个通道
        values = data_slice.flatten().tolist()
        values_str = ", ".join([str(value) for value in values])

        # 确保趋势计算的输入是 float
        input_tensor_for_trends = data_slice[:, 0].float()  # 假设趋势是基于第一个通道
        trends = torch.sum(torch.diff(input_tensor_for_trends)).item()
        trends_str = f"{trends:.0f}"

        # 模式相似度计算
        input_split_features = self._split_pattern(data_slice.cpu().numpy())  # numpy转换

        sim_scores = self._compute_similarity(input_split_features, pattern_numerical_array)
        probabilities = self._compute_probabilities(sim_scores)

        # 找到每个时间段最匹配的模式并组合
        all_top_categories = []
        all_top_probabilities = []
        all_top_patterns_str = []

        for p_idx in range(probabilities.shape[0]):  # 遍历每个子模式 (即每个3个时间步)
            prob = probabilities[p_idx]
            top_categories = np.argsort(prob)[::-1][:3]
            top_probabilities = prob[top_categories]

            # 从 descriptive_array 中获取对应的模式描述
            # 这里需要根据 pattern_descriptive_array 的具体结构来提取模式描述
            # 假设 pattern_descriptive_array 的每一行就是对应的模式 ID 或描述
            # 并且其形状可以被 top_categories 索引

            # !!! 这里的逻辑需要根据 pattern_descriptive_array 的实际内容进行调整 !!!
            # 例如：如果 pattern_descriptive_array[top_categories] 直接给出了文本，那就用
            # 如果是 ID，需要一个 ID 到文本的映射
            selected_top_patterns = [pattern_descriptive_array[cat_idx].flatten().tolist() for cat_idx in
                                     top_categories]  # 示例
            top_patterns_str = ", ".join([str(x) for x in selected_top_patterns])

            all_top_categories.extend(top_categories.tolist())
            all_top_probabilities.extend(top_probabilities.tolist())
            all_top_patterns_str.append(top_patterns_str)  # 将每个子模式的 top_patterns_str 收集起来

        # 拼接所有子模式的 top_patterns_str
        final_top_patterns_str = "; ".join(all_top_patterns_str)
        final_top_categories_str = ", ".join([str(x) for x in all_top_categories])
        final_top_probabilities_str = ", ".join([f"{x:.2f}" for x in all_top_probabilities])

        in_prompt = self._input_templates['PEMS']
        in_prompt = in_prompt.replace("value1, ..., valuen", values_str)
        in_prompt = in_prompt.replace("Trends", trends_str)
        in_prompt = in_prompt.replace("pattern1, ..., patternn", final_top_patterns_str)
        in_prompt = in_prompt.replace("top_categories", final_top_categories_str)
        in_prompt = in_prompt.replace("top_probabilities", final_top_probabilities_str)
        in_prompt = in_prompt.replace("[t1]", str(time_indices[0])).replace("[t2]", str(time_indices[-1]))

        tokenized_prompt = self.tokenizer.encode(in_prompt, return_tensors="pt", truncation=True,
                                                 max_length=self.gpt2_max_len).to(self.device)
        return tokenized_prompt

    def _gpt2_forward(self, tokenized_prompt):
        with torch.no_grad():
            prompt_embeddings = self.gpt2_model(tokenized_prompt).last_hidden_state
        return prompt_embeddings

    def generate_prompt(self, input_sequence, pattern_keys_tuple):
        """
        input_sequence: (batch, seq_len, num_nodes, dim)
        pattern_keys_tuple: 传入 GATBlockWithPrompt 的整个 self.pattern_keys 元组
        """
        batch_size, input_len, num_node, dim = input_sequence.shape

        tokenized_prompts_list = []
        max_token_count = 0

        for i in range(batch_size):
            for j in range(num_node):
                # 将 input_sequence[i, :, j] (单个节点序列) 传入 _prepare_single_prompt_tokenized
                # 确保这里是 torch.Tensor
                single_node_data_slice = input_sequence[i, :, j]
                tokenized_prompt = self._prepare_single_prompt_tokenized(single_node_data_slice, j, pattern_keys_tuple)
                max_token_count = max(max_token_count, tokenized_prompt.shape[1])
                tokenized_prompts_list.append((i, j, tokenized_prompt))  # 存储 batch_idx, node_idx 和 tokenized_prompt

        # 为所有 prompt 进行填充并一次性送入 GPT-2 (高效的做法)
        # 将所有 tokenized_prompts_list 中的 tokenized_prompt 拼接成一个大张量
        # 提取所有的 tokenized_prompt 텐서
        all_token_tensors = [item[2] for item in tokenized_prompts_list]

        # 填充
        padded_token_tensors = []
        attention_masks = []  # 也需要注意力掩码
        for tokens in all_token_tensors:
            pad_len = max_token_count - tokens.shape[1]
            padded_tokens = torch.cat(
                [tokens, torch.full((1, pad_len), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)],
                dim=1)
            padded_token_tensors.append(padded_tokens)

            # 生成注意力掩码
            attn_mask = torch.ones(1, max_token_count, dtype=torch.long, device=self.device)
            attn_mask[:, tokens.shape[1]:] = 0  # 填充部分为0
            attention_masks.append(attn_mask)

        # 批量处理 GPT-2
        batch_tokenized_prompts = torch.cat(padded_token_tensors, dim=0)  # (batch*num_nodes, max_token_count)
        batch_attention_masks = torch.cat(attention_masks, dim=0)  # (batch*num_nodes, max_token_count)

        with torch.no_grad():
            prompt_embeddings_batched = self.gpt2_model(input_ids=batch_tokenized_prompts,
                                                        attention_mask=batch_attention_masks).last_hidden_state
            # prompt_embeddings_batched: (batch*num_nodes, max_token_count, 768)

        # 提取最后一个 token 的嵌入
        last_token_embeddings_batched = prompt_embeddings_batched[:, -1, :]  # (batch*num_nodes, 768)

        # 将嵌入重塑回 (batch, num_node, 768)
        final_prompt_emb = last_token_embeddings_batched.view(batch_size, num_node, 768)

        return final_prompt_emb  # (batch, num_node, 768)

    def forward(self, x_raw_seq):
        """
        x_raw_seq: (batch, seq_len, num_nodes, dim)
        """
        batch_size, seq_len, num_nodes, dim = x_raw_seq.shape

        # ======= 生成 prompt embedding (使用内部的 GPT-2 逻辑) =======
        # generate_prompt 内部现在会使用 self.pattern_keys
        # generate_prompt 返回 (batch, num_node, 768)
        prompt_emb = self.generate_prompt(x_raw_seq, self.pattern_keys)

        # 扩展 prompt_emb 以匹配 seq_len，用于拼接
        # prompt_emb: (batch, num_nodes, 768)
        # 期望拼接维度：(batch, seq_len, num_nodes, 768)
        prompt_emb_expanded = prompt_emb.unsqueeze(1).repeat(1, seq_len, 1, 1)  # (batch, seq_len, num_nodes, 768)

        # 拼接 GPT2 embedding
        x_concat = torch.cat([x_raw_seq, prompt_emb_expanded], dim=-1)  # (batch, seq_len, num_nodes, dim + 768)

        # ======= 变换为 (batch*num_nodes, seq_len, dim_concat) 供 attention 使用 =======
        x_concat = x_concat.permute(0, 2, 1, 3).contiguous()  # (batch, num_nodes, seq_len, dim_concat)
        x_flat = x_concat.view(batch_size * num_nodes, seq_len, dim + 768)

        # ======= Attention =======
        x_attn, _ = self.attn(x_flat, x_flat, x_flat)  # (batch*num_nodes, seq_len, dim+768)

        # ======= 输出线性层 =======
        x_out = self.fc(x_attn)  # (batch*num_nodes, seq_len, dim_out)

        # reshape 回去
        x_out = x_out.view(batch_size, num_nodes, seq_len, -1).permute(0, 2, 1,
                                                                       3).contiguous()  # (batch, seq_len, num_nodes, dim_out)

        return x_out