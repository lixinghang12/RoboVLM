# import torch

# # 加载两个模型的state_dict
# path_1 = 'raw_model.pth'
# path_2 = 'new_model.pth'

# state_dict_1 = torch.load(path_1)
# state_dict_2 = torch.load(path_2)

# # 初始化结果字典
# comparison_results = {}

# # 遍历state_dict中的所有key
# for key in state_dict_1.keys():
#     # 确保两个state_dict都有这个key
#     if key in state_dict_2:
#         # 获取两个state_dict中相同key对应的权重
#         weights_1 = state_dict_1[key]
#         weights_2 = state_dict_2[key]
        
#         # 计算权重差异
#         diff = weights_1 - weights_2
        
#         # 计算差异的均值和标准差
#         mean_diff = diff.mean().item()
#         std_diff = diff.std().item()
#         if std_diff != 0:
#             print(key)
#             print(weights_1)
#             print(weights_2)
#             print('-'*100)
#         # 将结果保存到字典中
#         comparison_results[key] = {'mean_diff': mean_diff, 'std_diff': std_diff}

# # 打印或保存结果
# # print(comparison_results)

# # 如果需要保存为文件，可以使用以下代码
# import json

# with open('comparison_results.json', 'w') as f:
#     json.dump(comparison_results, f, indent=4)

import transformers
import torch

model = transformers.AutoModelForCausalLM.from_pretrained('/mnt/bn/robotics-data-lxh-lq/lxh/Qwen-VL', trust_remote_code=True, device_map='cuda:1')
tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/bn/robotics-data-lxh-lq/lxh/Qwen-VL', trust_remote_code=True)
images = torch.load('/mnt/bn/robotics-data-lxh-lq/RoboVLM/test.pt')
images = images[:2]
images = images.to('cuda:1')
with torch.no_grad():
    image_feat = model.transformer.visual(images)
from IPython import embed; embed()
