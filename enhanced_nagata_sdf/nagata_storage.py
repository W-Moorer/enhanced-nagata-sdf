"""
Nagata Enhance 数据存储模块
用于持久化存储裂隙边增强数据 (c_sharp)，避免重复计算

文件格式: .eng (Enhanced Nagata Geometry)
"""

import struct
import numpy as np
import os
from typing import Dict, Tuple, Optional


# 文件格式常量
ENG_MAGIC = b'ENG\x00'
ENG_VERSION = 1


def save_enhanced_data(filepath: str, c_sharps: Dict[Tuple[int, int], np.ndarray]) -> bool:
    """
    保存增强后的 Nagata 数据到 .eng 文件
    
    Args:
        filepath: 输出文件路径 (.eng)
        c_sharps: 裂隙边到共享系数的映射 {(v0, v1): np.array([cx, cy, cz])}
        
    Returns:
        bool: 保存是否成功
    """
    try:
        with open(filepath, 'wb') as f:
            # 1. 写入文件头 (16 bytes)
            f.write(ENG_MAGIC)                          # Magic (4 bytes)
            f.write(struct.pack('<I', ENG_VERSION))     # Version (4 bytes)
            f.write(struct.pack('<I', len(c_sharps)))   # NumCreaseEdges (4 bytes)
            f.write(struct.pack('<I', 0))               # Reserved (4 bytes)
            
            # 2. 写入数据块
            for edge_key, c_val in c_sharps.items():
                # 确保边键已排序
                v0, v1 = sorted(edge_key)
                # 写入顶点索引 (8 bytes)
                f.write(struct.pack('<II', v0, v1))
                # 写入 c_sharp 向量 (12 bytes)
                f.write(struct.pack('<fff', float(c_val[0]), float(c_val[1]), float(c_val[2])))
        
        print(f"已保存 {len(c_sharps)} 条裂隙边数据到: {filepath}")
        return True
        
    except Exception as e:
        print(f"保存失败: {e}")
        return False


def load_enhanced_data(filepath: str) -> Optional[Dict[Tuple[int, int], np.ndarray]]:
    """
    从 .eng 文件加载增强后的 Nagata 数据
    
    Args:
        filepath: 输入文件路径 (.eng)
        
    Returns:
        Dict 或 None: 裂隙边到共享系数的映射，加载失败则返回 None
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'rb') as f:
            # 1. 读取并验证文件头
            magic = f.read(4)
            if magic != ENG_MAGIC:
                print(f"无效的 ENG 文件: magic={repr(magic)}")
                return None
            
            version = struct.unpack('<I', f.read(4))[0]
            if version != ENG_VERSION:
                print(f"不支持的 ENG 版本: {version}")
                return None
            
            num_edges = struct.unpack('<I', f.read(4))[0]
            f.read(4)  # 跳过保留字段
            
            # 2. 读取数据块
            c_sharps = {}
            for _ in range(num_edges):
                data = f.read(20)  # 4 + 4 + 12
                if len(data) != 20:
                    print(f"数据不完整")
                    return None
                    
                v0, v1, cx, cy, cz = struct.unpack('<IIfff', data)
                c_sharps[(v0, v1)] = np.array([cx, cy, cz], dtype=np.float64)
            
            print(f"已加载 {len(c_sharps)} 条裂隙边数据从: {filepath}")
            return c_sharps
            
    except Exception as e:
        print(f"加载失败: {e}")
        return None


def get_eng_filepath(nsm_filepath: str) -> str:
    """
    根据 NSM 文件路径生成对应的 ENG 文件路径
    
    Args:
        nsm_filepath: NSM 文件路径
        
    Returns:
        str: 对应的 ENG 文件路径
    """
    base, _ = os.path.splitext(nsm_filepath)
    return base + '.eng'


def has_cached_data(nsm_filepath: str) -> bool:
    """
    检查是否存在缓存的增强数据
    
    Args:
        nsm_filepath: NSM 文件路径
        
    Returns:
        bool: 是否存在缓存
    """
    eng_path = get_eng_filepath(nsm_filepath)
    return os.path.exists(eng_path)


# 测试代码
if __name__ == '__main__':
    # 简单测试
    test_data = {
        (0, 1): np.array([0.1, 0.2, 0.3]),
        (1, 2): np.array([0.4, 0.5, 0.6]),
        (0, 2): np.array([0.7, 0.8, 0.9]),
    }
    
    test_file = 'test_output.eng'
    
    # 保存
    save_enhanced_data(test_file, test_data)
    
    # 加载
    loaded = load_enhanced_data(test_file)
    
    if loaded:
        print("加载成功:")
        for k, v in loaded.items():
            print(f"  {k}: {v}")
    
    # 清理
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"已删除测试文件: {test_file}")
