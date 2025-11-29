# from svgpathtools import svg2paths, Path as SVGPath, CubicBezier, Line, QuadraticBezier, Arc
import svgelements
import numpy as np
import os
from pathlib import Path
import random
from typing import List, Tuple, Optional
import io


def to_complex(p):
    """Helper to convert svgelements.Point/tuple to complex number"""
    if hasattr(p, 'x') and hasattr(p, 'y'):
        return complex(p.x, p.y)
    return complex(p[0], p[1])

def segment_to_cubics(seg) -> List[Tuple[complex, complex, complex, complex]]:
    """
    Updated to handle svgelements segments.
    svgelements converts everything to absolute coordinates automatically.
    """
    cls = seg.__class__.__name__

    # svgelements often returns standard geometric primitives
    if cls == "CubicBezier":
        return [(to_complex(seg.start),
                 to_complex(seg.control1),
                 to_complex(seg.control2),
                 to_complex(seg.end))]

    elif cls == "QuadraticBezier":
        p0 = to_complex(seg.start)
        q1 = to_complex(seg.control)
        p2 = to_complex(seg.end)
        # Exact conversion from Quadratic to Cubic
        c1 = p0 + (2/3)*(q1 - p0)
        c2 = p2 + (2/3)*(q1 - p2)
        return [(p0, c1, c2, p2)]

    elif cls == "Line":
        p0 = to_complex(seg.start)
        p1 = to_complex(seg.end)
        # Linear bezier: control points lie on the line
        c1 = p0 + (1/3)*(p1 - p0)
        c2 = p0 + (2/3)*(p1 - p0)
        return [(p0, c1, c2, p1)]

    elif cls == "Arc":
        # svgelements Arc objects can compute their own cubic approximation
        # We rely on the library's internal approximation if available,
        # or we can use the .d() path data to re-parse as cubics if needed.
        # Fortunately, svgelements Arcs are iterable as linear approximations
        # or can be converted. A robust way is usually provided by the library.
        # However, for deep learning simple approximations (Linear) are often safer
        # unless you specifically need the curve.
        # BETTER: svgelements paths usually treat arcs as specific segments.
        # We can ask it to output a CubicBezier approximation:
        cubics = []
        # svgelements doesn't have a direct 'as_cubic' on Arc, but the Path
        # iterator can be configured to reify Arcs.
        # For now, we treat it as a line from start to end to prevent crashes,
        # or you can implement the arc->cubic math.
        p0 = to_complex(seg.start)
        p1 = to_complex(seg.end)
        c1 = p0 + (1/3)*(p1 - p0)
        c2 = p0 + (2/3)*(p1 - p0)
        return [(p0, c1, c2, p1)]

    elif cls == "Move":
        return [] # Move commands don't produce renderable curves

    elif cls == "Close":
        # Handle 'Close' by drawing a line back to the start of the subpath
        # Note: You might need to track the subpath start manually if svgelements
        # doesn't provide it explicitly on the Close segment.
        # For simplicity in tensor representation, we often ignore explicit closes
        # or treat them as Lines if we know the points.
        return []

    return []

# ----------------------------
# 2) Extract subpaths as cubics
# ----------------------------
def svg_string_to_subpath_cubics(svg_string: str) -> List[Tuple[List[Tuple[complex, complex, complex, complex]], Tuple[float, float, float]]]:
    """
    Parses SVG string using svgelements to handle Group Transforms and Absolute Positioning.
    Returns a list of (subpath, fill_rgb), where each subpath is a list of cubic tuples,
    and fill_rgb is a tuple (r,g,b) in [0,1].
    """
    svg = svgelements.SVG.parse(svg_string)

    all_subpaths = []

    # Iterate over elements. svgelements flattens the structure.
    for element in svg.elements():
        if isinstance(element, svgelements.Path):
            # element is a Path object which is iterable yielding segments
            # The points in these segments are ALREADY transformed to screen space.

            current_subpath = []
            # Extract fill color
            fill = getattr(element, 'fill', None)
            if fill is None or fill == 'none':
                fill_rgb = (0.0, 0.0, 0.0)
            elif isinstance(fill, str):
                fill_rgb = hex_to_rgb(fill)
            else:
                # svgelements.Color object
                try:
                    fill_rgb = (fill.red / 255.0, fill.green / 255.0, fill.blue / 255.0)
                except Exception:
                    fill_rgb = (0.0, 0.0, 0.0)

            for seg in element:
                # Check if we hit a Move (new subpath)
                if isinstance(seg, svgelements.Move):
                    if current_subpath:
                        all_subpaths.append((current_subpath, fill_rgb))
                    current_subpath = []
                    continue

                # Convert segment to cubics
                cubics = segment_to_cubics(seg)
                current_subpath.extend(cubics)

            # Append the final subpath if it exists
            if current_subpath:
                all_subpaths.append((current_subpath, fill_rgb))

    return all_subpaths

# ----------------------------
# 3) Convert subpaths to tensor
# ----------------------------
def subpaths_to_tensor(subpaths: List[Tuple[List[Tuple[complex, complex, complex, complex]], Tuple[float, float, float]]],
                       max_paths: int, max_curves_per_path: int,
                       viewbox_size: int):
    """
    Returns:
        arr: (max_paths, max_curves_per_path+1, 4, 2)
        mask: (max_paths, max_curves_per_path+1) -> 1 for valid curves, 0 for padding
    """
    arr = np.zeros((max_paths, max_curves_per_path+1, 4, 2), dtype=np.float32)
    mask = np.zeros((max_paths, max_curves_per_path+1), dtype=np.float32)

    for i, (path, fill_rgb) in enumerate(subpaths[:max_paths]):
        n_curves = min(len(path), max_curves_per_path)
        for j in range(n_curves):
            p0, c1, c2, p3 = path[j]
            arr[i,j,0,0] = p0.real; arr[i,j,0,1] = p0.imag
            arr[i,j,1,0] = c1.real; arr[i,j,1,1] = c1.imag
            arr[i,j,2,0] = c2.real; arr[i,j,2,1] = c2.imag
            arr[i,j,3,0] = p3.real; arr[i,j,3,1] = p3.imag
            mask[i,j] = 1.0

        # Store fill color in the last "curve"
        arr[i, max_curves_per_path, 0, 0] = fill_rgb[0]
        arr[i, max_curves_per_path, 1, 0] = fill_rgb[1]
        arr[i, max_curves_per_path, 2, 0] = fill_rgb[2]
        # The rest remain zero
        mask[i, max_curves_per_path] = 1.0

    # Compute global bounding box across all subpaths (excluding fill row)
    valid = mask[:,:max_curves_per_path]==1
    all_points = arr[:,:max_curves_per_path][valid].reshape(-1,2)
    if all_points.shape[0] > 0:
        xs, ys = all_points[:,0], all_points[:,1]
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        if xmax - xmin < 1e-6: xmax = xmin + 1.0
        if ymax - ymin < 1e-6: ymax = ymin + 1.0
        arr[:,:max_curves_per_path,:,0] = (arr[:,:max_curves_per_path,:,0]-xmin)/(xmax-xmin) * viewbox_size
        arr[:,:max_curves_per_path,:,1] = (arr[:,:max_curves_per_path,:,1]-ymin)/(ymax-ymin) * viewbox_size

    return arr, mask

def canonicalize_svg(svg_string: str, max_paths: int, max_curves_per_path: int,
                    viewbox_size: int=128):
    subpaths = svg_string_to_subpath_cubics(svg_string)
    tensor, mask = subpaths_to_tensor(subpaths, max_paths, max_curves_per_path, viewbox_size)
    tensor[:,:-1] = tensor[:,:-1] / viewbox_size # Normalize curves to [0,1]
    # The fill color row is already in [0,1]
    return tensor, mask

def tensor_to_svg_string(tensor: np.ndarray, mask: np.ndarray,
                         width: int=128, height: int=128, stroke_width: int=1):
    """
    Convert (max_paths, max_curves_per_path+1, 4,2) tensor back to SVG.
    mask: indicates valid curves per path
    """
    svg_paths = []
    max_paths, max_curves_plus1, _, _ = tensor.shape
    max_curves = max_curves_plus1 - 1
    tensor = tensor.copy()
    tensor[:,:max_curves] = tensor[:,:max_curves] * width

    for i in range(max_paths):
        indices = np.where(mask[i,:max_curves] > 0.5)[0]
        if len(indices) == 0:
            continue

        # Get fill color from the last "curve"
        fill_r = tensor[i, max_curves, 0, 0]
        fill_g = tensor[i, max_curves, 1, 0]
        fill_b = tensor[i, max_curves, 2, 0]
        fill_hex = f"#{int(fill_r*255):02x}{int(fill_g*255):02x}{int(fill_b*255):02x}"

        parts = []
        first_j = indices[0]
        p0 = tensor[i, first_j, 0]
        parts.append(f"M {p0[0]:.2f} {p0[1]:.2f}")
        for j in indices:
            p0 = tensor[i, j, 0]
            c1 = tensor[i, j, 1]
            c2 = tensor[i, j, 2]
            p3 = tensor[i, j, 3]
            parts.append(
                f"C {c1[0]:.2f} {c1[1]:.2f}, "
                f"{c2[0]:.2f} {c2[1]:.2f}, "
                f"{p3[0]:.2f} {p3[1]:.2f}"
            )
        d_i = " ".join(parts)
        svg_paths.append(
            f'  <path d="{d_i}" fill="{fill_hex}" stroke="black" stroke-width="{stroke_width}"/>'
        )

    # wrap with full SVG
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'
    svg += "\n".join(svg_paths)
    svg += "\n</svg>"

    return svg

def hex_to_rgb(hex_color: str):
    """Convert #RRGGBB or #RGB to (r,g,b) in [0,1]"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    if len(hex_color) != 6:
        return (0.0, 0.0, 0.0)
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)
