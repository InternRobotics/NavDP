import numpy as np
import os
import yaml
import argparse
from pathlib import Path

def parse_yaml_file(file_path):
    """
    解析 YAML 文件，提取所有 start-end 对
    
    支持两种格式：
    1. 直接列表格式:
       - start: [...]
         end: [...]
    2. goal_pairs 键格式:
       goal_pairs:
       - start: [...]
         end: [...]
    
    返回: list of [start_x, start_y, goal_x, goal_y, goal_heading]
    """
    pairs = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
        
        # 如果 content 是字典，检查是否有 goal_pairs 键
        if isinstance(content, dict):
            # 优先查找 goal_pairs 键
            if 'goal_pairs' in content:
                goal_pairs_list = content['goal_pairs']
                if isinstance(goal_pairs_list, list):
                    content = goal_pairs_list  # 使用 goal_pairs 列表
                elif isinstance(goal_pairs_list, dict):
                    # 如果 goal_pairs 是单个字典
                    if 'start' in goal_pairs_list and 'end' in goal_pairs_list:
                        content = [goal_pairs_list]
            # 如果字典中有 start 和 end（单个条目）
            elif 'start' in content and 'end' in content:
                content = [content]
            # 如果字典中没有 goal_pairs，尝试直接处理
            else:
                # 可能是其他格式，尝试查找列表值
                for key, value in content.items():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], dict) and 'start' in value[0] and 'end' in value[0]:
                            content = value
                            break
        
        # 现在 content 应该是列表格式
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and 'start' in item and 'end' in item:
                    start = item['start']
                    end = item['end']
                    
                    # 提取坐标和朝向
                    start_x = float(start[0]) if len(start) > 0 else 0.0
                    start_y = float(start[1]) if len(start) > 1 else 0.0
                    start_heading = float(start[2]) if len(start) > 2 else 0.0
                    
                    end_x = float(end[0]) if len(end) > 0 else 0.0
                    end_y = float(end[1]) if len(end) > 1 else 0.0
                    end_heading = float(end[2]) if len(end) > 2 else 0.0
                    
                    # 格式: [start_x, start_y, goal_x, goal_y, goal_heading]
                    pairs.append([start_x, start_y, end_x, end_y, end_heading])
        
    except Exception as e:
        print(f"[WARNING] 解析文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    return pairs

def process_folder(folder_path, output_dir=None, output_suffix="pointgoal_pairs"):
    """
    处理文件夹中的所有 YAML 文件，为每个文件生成对应的 .npy 文件
    
    参数:
        folder_path: 输入文件夹路径
        output_dir: 输出文件夹路径（可选，默认与输入文件夹相同）
        output_suffix: 输出文件名后缀（可选，默认 "pointgoal_pairs"）
    
    返回:
        处理结果统计字典
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise ValueError(f"文件夹不存在: {folder_path}")
    
    if not folder_path.is_dir():
        raise ValueError(f"路径不是文件夹: {folder_path}")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = folder_path
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 YAML 文件
    yaml_files = list(folder_path.glob("*.yaml")) + list(folder_path.glob("*.yml"))
    
    if len(yaml_files) == 0:
        print(f"[WARNING] 文件夹 {folder_path} 中没有找到 YAML 文件")
        return {'total': 0, 'success': 0, 'failed': 0, 'files': []}
    
    print(f"[INFO] 找到 {len(yaml_files)} 个 YAML 文件")
    print(f"[INFO] 输出目录: {output_dir}")
    print("=" * 80)
    
    results = {
        'total': len(yaml_files),
        'success': 0,
        'failed': 0,
        'files': []
    }
    
    # 处理每个文件
    for yaml_file in sorted(yaml_files):
        print(f"\n[INFO] 处理文件: {yaml_file.name}")
        
        try:
            pairs = parse_yaml_file(yaml_file)
            
            if len(pairs) > 0:
                # 转换为 numpy 数组
                data = np.array(pairs, dtype=np.float64)
                
                # 生成输出文件名
                # 例如: xxx_goal_pairs.yaml -> xxx_pointgoal_pairs.npy
                # 或者: xxx.yaml -> xxx_pointgoal_pairs.npy
                base_name = yaml_file.stem  # 去掉扩展名
                
                # 如果文件名以 _goal_pairs 结尾，替换为 _pointgoal_pairs
                if base_name.endswith('_goal_pairs'):
                    output_name = base_name.replace('_goal_pairs', '_pointgoal_pairs') + '.npy'
                else:
                    # 否则添加后缀
                    output_name = f"{base_name}_{output_suffix}.npy"
                
                output_file = output_dir / output_name
                
                # 保存文件
                np.save(output_file, data)
                
                print(f"  -> 提取了 {len(pairs)} 个 start-end 对")
                print(f"  -> 已保存: {output_file.name}")
                print(f"  -> 数据形状: {data.shape}")
                
                results['success'] += 1
                results['files'].append({
                    'yaml_file': yaml_file.name,
                    'npy_file': output_file.name,
                    'pairs_count': len(pairs),
                    'status': 'success'
                })
            else:
                print(f"  -> 未找到有效的 start-end 对")
                results['failed'] += 1
                results['files'].append({
                    'yaml_file': yaml_file.name,
                    'npy_file': None,
                    'pairs_count': 0,
                    'status': 'failed'
                })
        
        except Exception as e:
            print(f"  -> [ERROR] 处理失败: {e}")
            results['failed'] += 1
            results['files'].append({
                'yaml_file': yaml_file.name,
                'npy_file': None,
                'pairs_count': 0,
                'status': 'error',
                'error': str(e)
            })
    
    # 打印总结
    print("\n" + "=" * 80)
    print("处理总结")
    print("=" * 80)
    print(f"总文件数: {results['total']}")
    print(f"成功: {results['success']}")
    print(f"失败: {results['failed']}")
    print(f"成功率: {results['success']/results['total']*100:.1f}%")
    print("=" * 80)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="从文件夹中的 YAML 文件生成对应的 .npy 文件（每个 YAML 文件生成一个 .npy 文件）")
    parser.add_argument(
        "--input_folder", 
        type=str, 
        required=True,
        help="包含 YAML 文件的输入文件夹路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出文件夹路径（默认: 与输入文件夹相同）"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="pointgoal_pairs",
        help="输出文件名后缀（默认: pointgoal_pairs，例如: xxx_goal_pairs.yaml -> xxx_pointgoal_pairs.npy）"
    )
    
    args = parser.parse_args()
    
    try:
        results = process_folder(args.input_folder, args.output_dir, args.output_suffix)
        print(f"\n[SUCCESS] 处理完成！")
        print(f"  - 成功处理: {results['success']} 个文件")
        print(f"  - 失败: {results['failed']} 个文件")
        
        if results['success'] > 0:
            total_pairs = sum(f['pairs_count'] for f in results['files'] if f['status'] == 'success')
            print(f"  - 总共生成: {total_pairs} 个点对")
        
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


#   # 步骤 1: 生成 NPY 文件
#   python test_yaml_data/chem_lab_gen_np.py \
#     --input_folder /home/pjlab/fbh/NavDP/test_yaml_data/goal_pairs_1_13

#   # 步骤 2: 将 NPY 文件复制到场景文件夹
#   python copy_npy_to_scenes.py \
#     --scene_root /home/pjlab/fbh/NavDP/scene_1231 \
#     --npy_source /home/pjlab/fbh/NavDP/test_yaml_data/goal_pairs_1_13

#   # 步骤 3: 批量测试（自动从 NPY 计算 num_episodes）
#   python batch_test_scenes.py --scene_indices 0,1,2