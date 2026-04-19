"""
对比单进程和并行版本的教师场生成结果。
"""

import numpy as np

def compare_results():
    """
    对比单进程和并行版本的输出结果。
    检查points数量、sdf数量，并在排序后比较数值差异。
    """
    print("=" * 60)
    print("对比单进程和并行结果（完整版）")
    print("=" * 60)
    
    single = np.load("outputs/test_single_full.npz", allow_pickle=True)
    parallel = np.load("outputs/test_parallel_full.npz", allow_pickle=True)
    
    single_points = single["points"]
    parallel_points = parallel["points"]
    
    print(f"单进程 points 数量: {single_points.shape[0]}")
    print(f"并行版 points 数量: {parallel_points.shape[0]}")
    print(f"points 数量一致: {single_points.shape[0] == parallel_points.shape[0]}")
    
    single_sdf = single["sdf"]
    parallel_sdf = parallel["sdf"]
    
    print(f"单进程 sdf 数量: {single_sdf.shape[0]}")
    print(f"并行版 sdf 数量: {parallel_sdf.shape[0]}")
    print(f"sdf 数量一致: {single_sdf.shape[0] == parallel_sdf.shape[0]}")
    
    if single_points.shape[0] > 0 and parallel_points.shape[0] > 0:
        if single_points.shape[0] == parallel_points.shape[0]:
            print("\n排序后比较...")
            
            single_sorted_idx = np.lexsort((single_points[:, 2], single_points[:, 1], single_points[:, 0]))
            parallel_sorted_idx = np.lexsort((parallel_points[:, 2], parallel_points[:, 1], parallel_points[:, 0]))
            
            single_points_sorted = single_points[single_sorted_idx]
            parallel_points_sorted = parallel_points[parallel_sorted_idx]
            
            single_sdf_sorted = single_sdf[single_sorted_idx]
            parallel_sdf_sorted = parallel_sdf[parallel_sorted_idx]
            
            points_diff = np.abs(single_points_sorted - parallel_points_sorted)
            sdf_diff = np.abs(single_sdf_sorted - parallel_sdf_sorted)
            
            print(f"points max diff: {np.max(points_diff)}")
            print(f"points mean diff: {np.mean(points_diff)}")
            print(f"sdf max diff: {np.max(sdf_diff)}")
            print(f"sdf mean diff: {np.mean(sdf_diff)}")
            print(f"sdf 差异接近 0: {np.mean(sdf_diff) < 1e-10}")
            
            single_nearest = single["nearest_points"]
            parallel_nearest = parallel["nearest_points"]
            single_nearest_sorted = single_nearest[single_sorted_idx]
            parallel_nearest_sorted = parallel_nearest[parallel_sorted_idx]
            nearest_diff = np.abs(single_nearest_sorted - parallel_nearest_sorted)
            
            print(f"nearest_points max diff: {np.max(nearest_diff)}")
            print(f"nearest_points mean diff: {np.mean(nearest_diff)}")
            
            single_normals = single["normals"]
            parallel_normals = parallel["normals"]
            single_normals_sorted = single_normals[single_sorted_idx]
            parallel_normals_sorted = parallel_normals[parallel_sorted_idx]
            normals_diff = np.abs(single_normals_sorted - parallel_normals_sorted)
            
            print(f"normals max diff: {np.max(normals_diff)}")
            print(f"normals mean diff: {np.mean(normals_diff)}")
    else:
        print("\n样本数为 0，无法进行数值比较。")
    
    print("\n" + "=" * 60)
    print("对比完成")
    print("=" * 60)


if __name__ == "__main__":
    compare_results()
