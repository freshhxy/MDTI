
import pandas as pd
import numpy as np
import torch
df=pd.read_csv('../data/PEMS/PeMS_lane_flow_40.csv')
print(df.head(0).columns[1:])

adj=pd.read_csv('../data/PEMS/PEMS_adj.csv',index_col=0)



# 定义传感器和车道的索引
indices = [
    'sensors 1 Lane 1', 'sensors 1 Lane 2', 'sensors 1 Lane 3',
    'sensors 1 Lane 4', 'sensors 1 Lane 5', 'sensors 2 Lane 1',
    'sensors 2 Lane 2', 'sensors 2 Lane 3', 'sensors 2 Lane 4',
    'sensors 2 Lane 5', 'sensors 3 Lane 1', 'sensors 3 Lane 2',
    'sensors 3 Lane 3', 'sensors 3 Lane 4', 'sensors 3 Lane 5',
    'sensors 4 Lane 1', 'sensors 4 Lane 2', 'sensors 4 Lane 3',
    'sensors 4 Lane 4', 'sensors 4 Lane 5', 'sensors 5 Lane 1',
    'sensors 5 Lane 2', 'sensors 5 Lane 3', 'sensors 5 Lane 4',
    'sensors 5 Lane 5', 'sensors 6 Lane 1', 'sensors 6 Lane 2',
    'sensors 6 Lane 3', 'sensors 6 Lane 4', 'sensors 6 Lane 5',
    'sensors 7 Lane 1', 'sensors 7 Lane 2', 'sensors 7 Lane 3',
    'sensors 7 Lane 4', 'sensors 7 Lane 5', 'sensors 8 Lane 1',
    'sensors 8 Lane 2', 'sensors 8 Lane 3', 'sensors 8 Lane 4',
    'sensors 8 Lane 5'
]

# 创建一个空的邻接矩阵
adj_matrix = pd.DataFrame(np.zeros((len(indices), len(indices))), index=indices, columns=indices, dtype=int)
print(np.array(adj_matrix.values))
for i in range(len(adj_matrix.values)) :
    for j in range(adj_matrix.values.shape[1]) :
        adj_matrix.values[i, j] = adj.values[i, j]



adj_matrix_df=adj_matrix


print('123',adj_matrix_df)


adjacent_lanes = {}

# 遍历每一行，找出相邻的车道（排除自身）
for lane in adj_matrix_df.index:
    # 获取当前车道所在行的数据
    row = adj_matrix_df.loc[lane]
    # 找出值为1且列名不等于当前车道的列（相邻的车道）
    adjacent = [col for col, value in row.items() if value == 1 and col != lane]
    # 存储相邻车道信息
    adjacent_lanes[lane] = adjacent

# 生成英文提示词
prompts = []
for lane, adj_lanes in adjacent_lanes.items():
    if adj_lanes:
        print(lane)
        print(adj_lanes)
        adj_lanes_str = ", ".join(adj_lanes)
        prompt = f"The lanes adjacent to {lane} are {adj_lanes_str}."
        prompts.append(prompt)
    else:
        prompt = f"There are no lanes adjacent to {lane}."
        prompts.append(prompt)

print(prompts)
"""


def forward(tokenized_prompt):
    with torch.no_grad():
        prompt_embeddings = model(tokenized_prompt).last_hidden_state
    return prompt_embeddings
#pad_token = '[PAD]'
#tokenizer.add_special_tokens({'pad_token': pad_token})
#tokenizer.pad_token = pad_token
tokenizer.pad_token = tokenizer.eos_token
tokenized_prompt=tokenizer(prompts,padding=True,return_tensors='pt').input_ids.to(device)

prompt_embeddings=forward(tokenized_prompt)
print(prompt_embeddings[:,-1,:].shape)

"""

