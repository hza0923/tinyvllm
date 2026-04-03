import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    # 将加载的权重张量（loaded_weight）原地复制到模型参数（param）的.data张量中
    # param已经在gpu中了，loaded_weight在cpu中，调用param.data.copy_(loaded_weight)会将loaded_weight从cpu复制到gpu，并覆盖param.data中的原有数据，从而完成权重的加载。
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    # qwen3.py模型定义中用来指定哪些权重名称需要进行特殊处理的映射关系。对于这些权重，加载器会根据映射关系找到对应的参数，并调用它们的weight_loader方法来加载权重。对于其他权重，则直接使用默认的加载方式。
    # packed_modules_mapping = {
    #     "q_proj": ("qkv_proj", "q"),
    #     "k_proj": ("qkv_proj", "k"),
    #     "v_proj": ("qkv_proj", "v"),
    #     "gate_proj": ("gate_up_proj", 0),
    #     "up_proj": ("gate_up_proj", 1),
    # }
    # print(f"Loading model from {path}...")
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {}) # 从模型中获取名为"packed_modules_mapping"的属性，如果该属性不存在，则返回一个空字典。
    for file in glob(os.path.join(path, "*.safetensors")): # glob(os.path.join(path, "*.safetensors"))会将path和"*.safetensors"拼接成一个完整的路径模式，然后glob函数会根据这个模式查找匹配的文件路径，并返回一个包含这些文件路径的列表。
        with safe_open(file, "pt", "cpu") as f: # 
            for weight_name in f.keys(): # f.keys()会返回safetensors文件中所有权重的名称列表。这个循环会遍历这些权重名称，逐个处理每个权重。
                # print(f"Loading weight: {weight_name}") 
                for k in packed_modules_mapping: # 遍历packed_modules_mapping中的每个键k，检查k是否是当前权重名称weight_name的子字符串。如果是，则说明这个权重需要进行特殊处理。
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k] # 从packed_modules_mapping中获取与k对应的值v和shard_id。v是该权重分片的新名字，shard_id是一个整数，用来指定当前权重所在的分片编号。
                        param_name = weight_name.replace(k, v) # 将权重名称weight_name中的k替换为v，得到新的参数名称param_name。这个新的参数名称是模型中对应的参数名称。
                        # print(f"  Found in packed_modules_mapping, loading to {param_name}, shard_id: {shard_id}")
                        # get_parameter方法会根据参数名称param_name从模型中获取对应的参数对象param。是模型中定义的一个nn.Parameter实例，代表了需要加载权重的参数。
                        # 以model.layers.0.self_attn.qkv_proj.weight为例：
                        #   先找model变量(Qwen3ForCausalLM)的self.model(Qwen3Model)
                        #   再找self.model的layers属性（一个nn.ModuleList）
                        #   再找layers中的第0个元素（Qwen3DecoderLayer）
                        #   再找这个Qwen3DecoderLayer的self_attn属性（Qwen3Attention）
                        #   再找self_attn的qkv_proj属性（QKVParallelLinear）
                        #   最后找qkv_proj的weight属性（一个nn.Parameter）。这个weight参数就是我们需要加载权重的目标参数。
                        param = model.get_parameter(param_name) 
                        weight_loader = getattr(param, "weight_loader") # 从参数对象param中获取名为"weight_loader"的属性，如果该属性不存在，则会引发AttributeError异常。这个weight_loader是一个方法，用于加载权重到参数中。
                        print(f"  Found in packed_modules_mapping, loading to {param_name}, shard_id: {shard_id}, weight_loader: {weight_loader.__qualname__}")
                        # 加载权重的本质：把从 .safetensors 文件中读取的张量（loaded_weight）复制到 param.data 中，让模型拥有预训练的权重值。
                        weight_loader(param, f.get_tensor(weight_name), shard_id) # 调用weight_loader方法来加载权重。这个方法接受三个参数：param是需要加载权重的参数对象，f.get_tensor(weight_name)会从safetensors文件中获取当前权重名称weight_name对应的权重张量，shard_id是分片编号，用于指定当前权重所在的分片。通过调用weight_loader方法，权重会被正确地加载到模型的参数中。
                        break
                # for else语句是Python中的一个特殊结构，它与for循环配合使用。当for循环正常结束（即没有被break语句中断）时，else块中的代码会被执行。
                # 此处的逻辑是：如果当前权重名称weight_name没有在packed_modules_mapping中找到匹配的键k（即没有被break语句中断），则执行else块中的代码，使用默认的加载方式来加载权重。
                else: 
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
        