"""
NSM文件读取器和可视化工具
基于 nsm_reader.hpp 格式定义
"""

import struct
import numpy as np
import pyvista as pv
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class NSMHeader:
    """NSM文件头结构 (64 bytes)"""
    magic: str           # "NSM\0"
    version: int         # 版本号
    num_vertices: int    # 顶点数量
    num_triangles: int   # 三角形数量
    reserved: bytes      # 保留字段 (48 bytes)


@dataclass
class NSMMeshData:
    """NSM网格数据结构"""
    vertices: np.ndarray           # [num_vertices, 3] 顶点坐标
    triangles: np.ndarray          # [num_triangles, 3] 三角形索引
    tri_face_ids: np.ndarray       # [num_triangles] 面片ID
    tri_vertex_normals: np.ndarray # [num_triangles, 3, 3] 每个三角形的3个顶点的法向量


def load_nsm(filepath: str) -> NSMMeshData:
    """
    读取NSM文件
    
    Args:
        filepath: NSM文件路径
        
    Returns:
        NSMMeshData: 网格数据对象
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误或版本不支持
    """
    with open(filepath, 'rb') as f:
        # 读取文件头 (64 bytes)
        header_data = f.read(64)
        if len(header_data) != 64:
            raise ValueError(f"文件头不完整: 期望64字节, 实际{len(header_data)}字节")
        
        # 解析文件头
        magic = header_data[0:4].decode('ascii', errors='ignore')
        version = struct.unpack('<I', header_data[4:8])[0]
        num_vertices = struct.unpack('<I', header_data[8:12])[0]
        num_triangles = struct.unpack('<I', header_data[12:16])[0]
        reserved = header_data[16:64]
        
        # 验证magic
        if magic != 'NSM\x00':
            raise ValueError(f"无效的magic标识: {repr(magic)}, 期望: 'NSM\\x00'")
        
        # 验证版本
        if version != 1:
            raise ValueError(f"不支持的版本: {version}, 期望: 1")
        
        print(f"NSM文件信息:")
        print(f"  版本: {version}")
        print(f"  顶点数: {num_vertices}")
        print(f"  三角形数: {num_triangles}")
        
        # 读取顶点数据 [num_vertices * 3] double
        vertices = np.fromfile(f, dtype=np.float64, count=num_vertices * 3)
        if len(vertices) != num_vertices * 3:
            raise ValueError(f"顶点数据不完整: 期望{num_vertices * 3}个值, 实际{len(vertices)}个")
        vertices = vertices.reshape(num_vertices, 3)
        
        # 读取三角形索引 [num_triangles * 3] uint32
        triangles = np.fromfile(f, dtype=np.uint32, count=num_triangles * 3)
        if len(triangles) != num_triangles * 3:
            raise ValueError(f"三角形索引不完整: 期望{num_triangles * 3}个值, 实际{len(triangles)}个")
        triangles = triangles.reshape(num_triangles, 3)
        
        # 读取面片ID [num_triangles] uint32
        tri_face_ids = np.fromfile(f, dtype=np.uint32, count=num_triangles)
        if len(tri_face_ids) != num_triangles:
            raise ValueError(f"面片ID不完整: 期望{num_triangles}个值, 实际{len(tri_face_ids)}个")
        
        # 读取顶点法向量 [num_triangles * 3 * 3] double
        # 每个三角形有3个顶点，每个顶点有3个法向量分量
        tri_vertex_normals = np.fromfile(f, dtype=np.float64, count=num_triangles * 3 * 3)
        if len(tri_vertex_normals) != num_triangles * 3 * 3:
            raise ValueError(f"法向量数据不完整: 期望{num_triangles * 3 * 3}个值, 实际{len(tri_vertex_normals)}个")
        tri_vertex_normals = tri_vertex_normals.reshape(num_triangles, 3, 3)
        
        return NSMMeshData(
            vertices=vertices,
            triangles=triangles,
            tri_face_ids=tri_face_ids,
            tri_vertex_normals=tri_vertex_normals
        )


def get_nsm_info(filepath: str) -> Tuple[int, int]:
    """
    获取NSM文件信息（不读取全部数据）
    
    Args:
        filepath: NSM文件路径
        
    Returns:
        Tuple[int, int]: (顶点数, 三角形数)
    """
    with open(filepath, 'rb') as f:
        header_data = f.read(64)
        if len(header_data) != 64:
            raise ValueError(f"文件头不完整")
        
        magic = header_data[0:4].decode('ascii', errors='ignore')
        version = struct.unpack('<I', header_data[4:8])[0]
        num_vertices = struct.unpack('<I', header_data[8:12])[0]
        num_triangles = struct.unpack('<I', header_data[12:16])[0]
        
        if magic != 'NSM\x00' or version != 1:
            raise ValueError("无效的NSM文件")
        
        return num_vertices, num_triangles


def create_pyvista_mesh(mesh_data: NSMMeshData) -> pv.PolyData:
    """
    从NSM数据创建PyVista网格
    
    Args:
        mesh_data: NSM网格数据
        
    Returns:
        pv.PolyData: PyVista网格对象
    """
    # 创建点云
    points = mesh_data.vertices
    
    # 创建面片 (PyVista需要每个面片前面加上点的数量)
    faces = np.hstack([
        np.full((mesh_data.triangles.shape[0], 1), 3, dtype=np.int32),
        mesh_data.triangles.astype(np.int32)
    ]).flatten()
    
    # 创建PolyData
    mesh = pv.PolyData(points, faces)
    
    # 添加面片ID作为单元数据
    mesh.cell_data['face_id'] = mesh_data.tri_face_ids
    
    return mesh


def visualize_nsm(filepath: str, 
                  show_normals: bool = True,
                  normal_scale: float = 0.01,
                  normal_skip: int = 1,
                  show_edges: bool = True,
                  color_by_face_id: bool = False):
    """
    可视化NSM文件（模型+法向量）
    
    Args:
        filepath: NSM文件路径
        show_normals: 是否显示法向量
        normal_scale: 法向量箭头缩放比例
        normal_skip: 跳过的法向量数量（用于减少显示密度）
        show_edges: 是否显示网格边
        color_by_face_id: 是否按面片ID着色
    """
    # 加载数据
    print(f"加载文件: {filepath}")
    mesh_data = load_nsm(filepath)
    
    # 创建PyVista网格
    mesh = create_pyvista_mesh(mesh_data)
    
    # 创建绘图器
    plotter = pv.Plotter()
    plotter.set_background('white')
    plotter.add_axes()
    plotter.add_bounding_box()
    
    # 添加网格
    if color_by_face_id:
        # 按面片ID着色
        mesh_actor = plotter.add_mesh(
            mesh, 
            scalars='face_id',
            show_edges=show_edges,
            cmap='tab20',
            opacity=0.9
        )
        plotter.add_scalar_bar(title='Face ID')
    else:
        # 使用默认颜色
        mesh_actor = plotter.add_mesh(
            mesh, 
            color='lightblue',
            show_edges=show_edges,
            opacity=0.9
        )
    
    # 显示法向量
    if show_normals:
        print(f"添加法向量显示 (每{normal_skip}个三角形显示一次)...")
        
        # 计算每个三角形顶点的世界坐标和法向量
        arrow_centers = []
        arrow_directions = []
        
        for tri_idx in range(0, mesh_data.triangles.shape[0], normal_skip):
            tri = mesh_data.triangles[tri_idx]
            for local_vert_idx in range(3):
                # 获取顶点索引
                vert_idx = tri[local_vert_idx]
                # 获取顶点坐标
                vertex_pos = mesh_data.vertices[vert_idx]
                # 获取法向量
                normal = mesh_data.tri_vertex_normals[tri_idx, local_vert_idx]
                
                arrow_centers.append(vertex_pos)
                arrow_directions.append(normal)
        
        arrow_centers = np.array(arrow_centers)
        arrow_directions = np.array(arrow_directions)
        
        # 创建箭头glyph
        if len(arrow_centers) > 0:
            # 创建箭头数据源
            arrow = pv.Arrow()
            
            # 创建点云并设置向量
            arrows_poly = pv.PolyData(arrow_centers)
            arrows_poly['vectors'] = arrow_directions * normal_scale
            
            # 使用glyph创建箭头
            glyphs = arrows_poly.glyph(
                orient='vectors',
                scale='vectors',
                factor=1.0,
                geom=arrow
            )
            
            plotter.add_mesh(glyphs, color='red', opacity=0.8)
            print(f"  显示了 {len(arrow_centers)} 个法向量")
    
    # 添加标题
    plotter.add_title(
        f"NSM Model: {mesh_data.triangles.shape[0]} triangles, "
        f"{mesh_data.vertices.shape[0]} vertices",
        font_size=12
    )
    
    print("\n交互式可视化已启动:")
    print("  - 左键拖动: 旋转")
    print("  - 右键拖动: 缩放")
    print("  - 中键拖动: 平移")
    print("  - 滚轮: 缩放")
    print("  - 'q': 退出")
    
    # 启动交互式窗口
    plotter.show()


def main():
    """主函数"""
    import sys
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python nsm_reader.py <nsm文件路径> [选项]")
        print("\n选项:")
        print("  --no-normals       不显示法向量")
        print("  --normal-scale N   设置法向量缩放比例 (默认: 0.01)")
        print("  --normal-skip N    设置法向量显示密度 (默认: 1)")
        print("  --no-edges         不显示网格边")
        print("  --color-by-id      按面片ID着色")
        print("\n示例:")
        print("  python nsm_reader.py model.nsm")
        print("  python nsm_reader.py model.nsm --normal-scale 0.05 --normal-skip 5")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # 解析选项
    show_normals = '--no-normals' not in sys.argv
    normal_scale = 0.01
    normal_skip = 1
    show_edges = '--no-edges' not in sys.argv
    color_by_face_id = '--color-by-id' in sys.argv
    
    # 解析数值参数
    for i, arg in enumerate(sys.argv):
        if arg == '--normal-scale' and i + 1 < len(sys.argv):
            normal_scale = float(sys.argv[i + 1])
        elif arg == '--normal-skip' and i + 1 < len(sys.argv):
            normal_skip = int(sys.argv[i + 1])
    
    # 运行可视化
    visualize_nsm(
        filepath,
        show_normals=show_normals,
        normal_scale=normal_scale,
        normal_skip=normal_skip,
        show_edges=show_edges,
        color_by_face_id=color_by_face_id
    )


if __name__ == '__main__':
    main()
