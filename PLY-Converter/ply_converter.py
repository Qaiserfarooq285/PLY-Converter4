#!/usr/bin/env python3
"""
Enhanced PLY Converter with Robust Fallback Methods

This version includes:
- Multiple fallback methods when Open3D is not available
- Better error handling and logging
- Simplified smoothing algorithms that work with basic libraries
- Point cloud to mesh conversion using multiple approaches

Dependencies (required):
  pip install trimesh numpy plyfile scipy

Dependencies (optional for advanced features):
  pip install open3d scikit-image
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement

# Optional dependencies - graceful fallback if not available
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    o3d = None
    HAS_OPEN3D = False

try:
    from scipy.spatial import cKDTree, distance
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    cKDTree = None
    ndimage = None
    HAS_SCIPY = False

try:
    from skimage import measure as skmeasure
    HAS_SKIMAGE = True
except ImportError:
    skmeasure = None
    HAS_SKIMAGE = False


def log(msg: str) -> None:
    """Simple logging function"""
    print(f"[PLY-Converter] {msg}", flush=True)


def create_mesh_from_points_basic(points: np.ndarray, colors: Optional[np.ndarray] = None) -> trimesh.Trimesh:
    """
    Create a basic mesh from points using simple triangulation
    This is a fallback method when advanced reconstruction is not available
    """
    try:
        # Use trimesh's convex hull as a simple mesh creation method
        mesh = trimesh.Trimesh(vertices=points).convex_hull
        
        if colors is not None and len(colors) == len(points):
            # Map colors to mesh vertices by finding closest points
            mesh_vertices = np.asarray(mesh.vertices)
            if HAS_SCIPY and cKDTree is not None:
                tree = cKDTree(points)
                distances, indices = tree.query(mesh_vertices, k=1)
                mesh_colors = colors[indices]
            else:
                # Simple closest point mapping without scipy
                mesh_colors = []
                for mv in mesh_vertices:
                    dists = np.sum((points - mv)**2, axis=1)
                    closest_idx = np.argmin(dists)
                    mesh_colors.append(colors[closest_idx])
                mesh_colors = np.array(mesh_colors)
            
            # Set vertex colors
            if mesh_colors.max() <= 1.0:
                mesh_colors = (mesh_colors * 255).astype(np.uint8)
            mesh.visual.vertex_colors = mesh_colors
        
        log(f"Created basic mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh
        
    except Exception as e:
        log(f"Basic mesh creation failed: {e}")
        # Ultimate fallback - create a simple box
        return trimesh.creation.box(extents=[1, 1, 1])


def load_ply_file(file_path: str) -> Dict[str, Any]:
    """Load PLY file and extract vertices, colors, normals"""
    try:
        # Try with trimesh first (most compatible)
        mesh_data = trimesh.load(str(file_path))
        
        if hasattr(mesh_data, 'vertices'):
            vertices = np.asarray(mesh_data.vertices)
            
            # Extract colors if available
            colors = None
            if hasattr(mesh_data.visual, 'vertex_colors'):
                colors = np.asarray(mesh_data.visual.vertex_colors)
                if colors.shape[1] == 4:  # RGBA to RGB
                    colors = colors[:, :3]
                if colors.max() > 1.0:  # Convert to 0-1 range
                    colors = colors / 255.0
            
            # Extract faces if it's a mesh
            faces = None
            if hasattr(mesh_data, 'faces') and len(mesh_data.faces) > 0:
                faces = np.asarray(mesh_data.faces)
            
            return {
                'vertices': vertices,
                'faces': faces,
                'colors': colors,
                'is_point_cloud': faces is None or len(faces) == 0
            }
    
    except Exception as e:
        log(f"Trimesh loading failed: {e}")
    
    try:
        # Fallback to plyfile
        plydata = PlyData.read(file_path)
        vertices = plydata['vertex']
        
        # Extract coordinates
        coords = np.column_stack([vertices['x'], vertices['y'], vertices['z']])
        
        # Extract colors if available
        colors = None
        if 'red' in vertices.dtype.names and 'green' in vertices.dtype.names and 'blue' in vertices.dtype.names:
            colors = np.column_stack([vertices['red'], vertices['green'], vertices['blue']])
            if colors.max() > 1.0:
                colors = colors / 255.0
        
        # Check for faces
        faces = None
        if 'face' in plydata:
            face_data = plydata['face']
            if 'vertex_indices' in face_data.dtype.names:
                faces = np.array([list(face[0]) for face in face_data['vertex_indices']])
        
        return {
            'vertices': coords,
            'faces': faces,
            'colors': colors,
            'is_point_cloud': faces is None or len(faces) == 0
        }
        
    except Exception as e:
        log(f"PLYfile loading failed: {e}")
        raise RuntimeError(f"Failed to load PLY file: {e}")


def smooth_mesh_basic(mesh: trimesh.Trimesh, smoothing_level: str = "medium") -> trimesh.Trimesh:
    """Apply basic smoothing using trimesh methods"""
    try:
        iterations_map = {
            'light': 1,
            'medium': 2,
            'high': 3,
            'ultra': 5
        }
        
        iterations = iterations_map.get(smoothing_level, 2)
        
        smoothed_mesh = mesh
        for i in range(iterations):
            if hasattr(smoothed_mesh, 'smoothed'):
                smoothed_mesh = smoothed_mesh.smoothed()
            else:
                break
        
        log(f"Applied basic smoothing: {iterations} iterations")
        return smoothed_mesh
        
    except Exception as e:
        log(f"Basic smoothing failed: {e}")
        return mesh


def open3d_reconstruction(vertices: np.ndarray, colors: Optional[np.ndarray] = None, smoothing_level: str = "medium") -> Optional[trimesh.Trimesh]:
    """Advanced reconstruction using Open3D (if available)"""
    if not HAS_OPEN3D:
        return None
    
    try:
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals
        bbox = vertices.max(axis=0) - vertices.min(axis=0)
        radius = max(np.linalg.norm(bbox) * 0.02, 0.01)
        
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
        pcd.normalize_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        
        # Poisson reconstruction
        depth_map = {'light': 8, 'medium': 9, 'high': 10, 'ultra': 11}
        depth = depth_map.get(smoothing_level, 9)
        
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.1, linear_fit=False
        )
        
        # Clean up low-density vertices
        if len(densities) > 0:
            densities = np.asarray(densities)
            thresh = np.quantile(densities, 0.01)  # Remove bottom 1%
            vertices_to_remove = densities < thresh
            mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Apply smoothing
        smoothing_iterations = {'light': 3, 'medium': 5, 'high': 10, 'ultra': 15}
        iterations = smoothing_iterations.get(smoothing_level, 5)
        
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations, lambda_filter=0.5)
        
        # Clean up
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        
        # Convert to trimesh
        vertices_np = np.asarray(mesh.vertices)
        faces_np = np.asarray(mesh.triangles)
        
        tm = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
        
        # Transfer colors if available
        if mesh.has_vertex_colors():
            vertex_colors = np.asarray(mesh.vertex_colors)
            if vertex_colors.max() <= 1.0:
                vertex_colors = (vertex_colors * 255).astype(np.uint8)
            tm.visual.vertex_colors = vertex_colors
        
        log(f"Open3D reconstruction successful: {len(tm.vertices)} vertices, {len(tm.faces)} faces")
        return tm
        
    except Exception as e:
        log(f"Open3D reconstruction failed: {e}")
        return None


class PLYConverter:
    """Enhanced PLY Converter with robust fallback methods"""
    
    def convert_ply(self, input_path: str, output_dir: str, output_formats: list, 
                   conversion_id: str, progress_callback: Optional[Callable] = None, 
                   smoothing_level: str = "medium") -> Dict[str, str]:
        """Convert PLY file to specified formats with smoothing"""
        
        def update_progress(message: str, progress: int):
            if progress_callback:
                progress_callback(message, progress)
            log(f"Progress {progress}%: {message}")
        
        try:
            update_progress("Loading PLY file...", 5)
            
            # Load PLY data
            ply_data = load_ply_file(input_path)
            vertices = ply_data['vertices']
            faces = ply_data['faces']
            colors = ply_data['colors']
            is_point_cloud = ply_data['is_point_cloud']
            
            update_progress(f"Loaded {len(vertices)} vertices", 15)
            
            if is_point_cloud:
                update_progress("Point cloud detected, reconstructing surface...", 25)
                
                # Try Open3D reconstruction first
                mesh = open3d_reconstruction(vertices, colors, smoothing_level)
                
                if mesh is None:
                    update_progress("Using basic surface reconstruction...", 35)
                    mesh = create_mesh_from_points_basic(vertices, colors)
                else:
                    update_progress("Advanced surface reconstruction completed", 45)
            else:
                update_progress("Mesh detected, processing...", 25)
                # Create trimesh from existing mesh data
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                if colors is not None:
                    if colors.max() <= 1.0:
                        colors = (colors * 255).astype(np.uint8)
                    mesh.visual.vertex_colors = colors
            
            update_progress(f"Applying {smoothing_level} smoothing...", 55)
            
            # Apply smoothing (basic method if Open3D not available)
            if not HAS_OPEN3D or is_point_cloud:
                mesh = smooth_mesh_basic(mesh, smoothing_level)
            
            update_progress("Cleaning up mesh...", 70)
            
            # Basic cleanup
            try:
                mesh.remove_degenerate_faces()
                mesh.remove_duplicate_faces()
                mesh.remove_unreferenced_vertices()
                mesh.fill_holes()
            except Exception as e:
                log(f"Mesh cleanup warning: {e}")
            
            update_progress("Exporting files...", 80)
            
            # Export to specified formats
            results = {}
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for i, fmt in enumerate(output_formats):
                try:
                    progress = 80 + int((i + 1) / len(output_formats) * 15)
                    update_progress(f"Exporting {fmt.upper()}...", progress)
                    
                    filename = f"{conversion_id}_smooth.{fmt}"
                    file_path = output_path / filename
                    
                    # Export based on format
                    mesh.export(str(file_path))
                    results[fmt] = str(file_path)
                    
                    log(f"Exported {fmt.upper()}: {file_path}")
                    
                except Exception as export_error:
                    log(f"Failed to export {fmt}: {export_error}")
                    continue
            
            if not results:
                raise RuntimeError("No files were successfully exported")
            
            update_progress("Conversion completed successfully!", 100)
            return results
            
        except Exception as e:
            error_msg = f"Conversion failed: {str(e)}"
            log(f"ERROR: {error_msg}")
            log(f"Traceback: {traceback.format_exc()}")
            update_progress(error_msg, 0)
            raise RuntimeError(error_msg) from e


def main():
    """Simple test function"""
    print("PLY Converter - Enhanced Version")
    print(f"Open3D available: {HAS_OPEN3D}")
    print(f"SciPy available: {HAS_SCIPY}")
    print(f"Scikit-image available: {HAS_SKIMAGE}")
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if os.path.exists(input_file):
            converter = PLYConverter()
            try:
                results = converter.convert_ply(
                    input_file, 
                    "output", 
                    ['stl', 'obj'], 
                    'test',
                    lambda msg, prog: print(f"[{prog}%] {msg}"),
                    'medium'
                )
                print("Conversion successful!")
                for fmt, path in results.items():
                    print(f"  {fmt.upper()}: {path}")
            except Exception as e:
                print(f"Conversion failed: {e}")
        else:
            print(f"File not found: {input_file}")
    else:
        print("Usage: python ply_converter.py <input.ply>")


if __name__ == "__main__":
    main()