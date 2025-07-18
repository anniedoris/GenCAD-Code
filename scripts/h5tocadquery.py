import glob
import h5py
import numpy as np
from geom_utils import *
import os
from deepcad_constants import *
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import argparse

###########
# May need to update this if your data is stored elsewhere
H5_VEC_FOLDER = 'deepcad_derived/data/cad_vec'
UNQUANTIZE = True # TODO: support unquantized?
generate_stls = True
###########

def extract_h5_file(h5_file_path):
    """
    Takes in an h5 file path and returns a numpy array of the vectorized CAD command

    Args:
        h5_file_path (str): path to h5 file (.h5)

    Returns:
        data (numpyarray): of the CAD command sequence, per DeepCAD tokenization
    """
    with h5py.File(h5_file_path, "r") as h5_file:
        dataset_name = list(h5_file.keys())[0]  # Example: Get the first dataset
        data = h5_file[dataset_name][()]  # Read dataset into memory
    return data

def create_save_dir(vec_dir):
    """
    Sets up a directory called cadquery for saving generated python files. It will create it one level up from the H5_VEC_FOLDER.

    Args:
        vec_dir (str): path to h5 vecs/ folder

    Returns:
        cadquery_data (str): returns path to newly generated cadquery/ directory
    """
    root_data = vec_dir.rsplit("/", 1)[0]
    cadquery_data = root_data + "/cadquery"
    if not os.path.exists(cadquery_data):
        os.makedirs(cadquery_data) 
        
    # Also create a logsdir
    if not os.path.exists(root_data + "/logs"):
        os.makedirs(root_data + "/logs")
    return cadquery_data
    
def convert_h5_to_cadquery(vecs, save_python_dir, save_step_path, use_fixed_decimal, truncate):
    """
    Master function that converts tokenized numpy array of CAD commands into a CADQuery python script

    Args:
        vecs (numpyarray): array containing the CAD commands
        save_python_dir (str): full path to where the .py file should be saved

    Returns:
        None
    """ 
    
    def split_by_sketches(arr):
        # Finding indices where the first element is 5
        split_indices = np.where(arr[:, 0] == EXTRUDE)[0] + 1
        # Splitting the array
        split_arrays = np.split(arr, split_indices)[:-1]
        # Splitting the arrays
        sketches = [arr[:-1] for arr in split_arrays]  # Remove last row
        extrudes = [arr[-1:] for arr in split_arrays]  # Keep only the last row
        return sketches, extrudes
    
    def split_by_loops(arr):
        # Finding indices where the first element is 4
        split_indices = np.where(arr[:, 0] == SOL)[0]
        # Splitting the array
        split_arrays = np.split(arr, split_indices)[1:]
        loops = [arr[1:] for arr in split_arrays]
        return loops
        
    def extract_extrude_params(extrusion_command, unquantize):
        extrude_theta = extrusion_command[0][6]
        extrude_phi = extrusion_command[0][7]
        extrude_gamma = extrusion_command[0][8]
        extrude_px = extrusion_command[0][9]
        extrude_py = extrusion_command[0][10]
        extrude_pz = extrusion_command[0][11]
        extrude_scale = extrusion_command[0][12]
        extrude_dir1 = extrusion_command[0][13]
        extrude_dir2 = extrusion_command[0][14]
        extrude_op = int(extrusion_command[0][15])
        extrude_type = int(extrusion_command[0][16])
        if unquantize:
            extrude_dir1 = extrude_dir1 /256 * 2 - 1.0
            extrude_dir2 = extrude_dir2 /256 * 2 - 1.0
            extrude_scale = extrude_scale / 256 * 2
        return extrude_theta, extrude_phi, extrude_gamma, extrude_px, extrude_py, extrude_pz, extrude_scale, extrude_dir1, extrude_dir2, extrude_op, extrude_type
    
    
    def cadquery_workplane(sketch_plane_obj, sketch_num):
        workplane_comment = f"# Generating a workplane for sketch {sketch_num}\n"
        # if any(-1.0 < coord <= 0.0 for coord in sketch_plane_obj.origin[0]):
        #     if any(coord == 1.0 for coord in sketch_plane_obj.origin[1]) and (any(coord == 1.0 for coord in sketch_plane_obj.origin[2])):
        #         plane = "front"
        #         python_command = f"wp_sketch{sketch_num} = cq.Workplane(plane)\n"
        # else:
        if args.use_fixed_decimal is True: 
            python_command = f"wp_sketch{sketch_num} = cq.Workplane(cq.Plane(cq.Vector({sketch_plane_obj.origin[0]:.{args.decimal_points}f}, {sketch_plane_obj.origin[1]:.{args.decimal_points}f}, {sketch_plane_obj.origin[2]:.{args.decimal_points}f}), cq.Vector({sketch_plane_obj.x_axis[0]:.{args.decimal_points}f}, {sketch_plane_obj.x_axis[1]:.{args.decimal_points}f}, {sketch_plane_obj.x_axis[2]:.{args.decimal_points}f}), cq.Vector({sketch_plane_obj.normal[0]:.{args.decimal_points}f}, {sketch_plane_obj.normal[1]:.{args.decimal_points}f}, {sketch_plane_obj.normal[2]:.{args.decimal_points}f})))\n"
        else:
            python_command = f"wp_sketch{sketch_num} = cq.Workplane(cq.Plane(cq.Vector({sketch_plane_obj.origin[0]}, {sketch_plane_obj.origin[1]}, {sketch_plane_obj.origin[2]}), cq.Vector({sketch_plane_obj.x_axis[0]}, {sketch_plane_obj.x_axis[1]}, {sketch_plane_obj.x_axis[2]}), cq.Vector({sketch_plane_obj.normal[0]}, {sketch_plane_obj.normal[1]}, {sketch_plane_obj.normal[2]})))\n"
        # elif any(1.0 < coord <= 0.0 for coord in sketch_plane_obj.origin[0]):
        #     if any(coord == 1.0 for coord in sketch_plane_obj.origin[1]) and (any(coord == 1.0 for coord in sketch_plane_obj.origin[2])):
        #         plane = 'XY'
        # elif any(-1.0 < coord <= 0.0 for coord in sketch_plane_obj.origin[0]):
        #     if any(coord == 1.0 for coord in sketch_plane_obj.origin[1]) and (any(coord == 1.0 for coord in sketch_plane_obj.origin[2])):
        #         plane = 'XY'
        return workplane_comment + python_command
    
    def cadquery_line(x, y, curr_x, curr_y, loop_list, unquantize, extrude_scale):
        if (x == curr_x) and (y == curr_y): # handles case where there are duplicate points or first point is zero and it also closes at zero, need to handle case of first point and last point are same but not zero
            print(f"problem: {x}, {y}, {curr_x}, {curr_y}")
            return ""
        else:
            if unquantize:
                scale = extrude_scale / (256 / 2 * NORM_FACTOR - 1)
                translate = -(256 / 2)
                x = (x + translate)*scale
                y = (y + translate)*scale
            if len(loop_list) == 0:
                return f".moveTo({x:.{args.decimal_point}f}, {y:.{args.decimal_points}f})" if args.use_fixed_decimal else f".moveTo({x}, {y})"
            else:
                return f".lineTo({x:.{args.decimal_points}f}, {y:.{args.decimal_points}f})" if args.use_fixed_decimal else f".lineTo({x}, {y})"
        
    def cadquery_arc(x, y, curr_x, curr_y, sweep, dir_flag, loop_list, unquantize, extrude_scale):
        arc_out = get_arc(x, y, curr_x, curr_y, sweep, dir_flag, is_numerical=True)
        if arc_out == None: # handles the case where there really isn't an arc, not different from previous location
            return ""
        else:
            start_point, mid_point, end_point = arc_out
        if unquantize:
            scale = extrude_scale / (256 / 2 * NORM_FACTOR - 1)
            translate = -(256 / 2)
            start_point_x = (start_point[0] + translate) * scale
            start_point_y = (start_point[1] + translate) * scale
            mid_point_x = (mid_point[0] + translate) * scale
            mid_point_y = (mid_point[1] + translate) * scale
            end_point_x = (end_point[0] + translate) * scale
            end_point_y = (end_point[1] + translate) * scale
            curr_x = (curr_x + translate) * scale
            curr_y = (curr_y + translate) * scale
            
        if len(loop_list) == 0:
            if args.use_fixed_decimal is True:
                return f".moveTo({curr_x:.{args.decimal_points}f}, {curr_y:.{args.decimal_points}f}).threePointArc(({mid_point_x:.{args.decimal_points}f}, {mid_point_y:.{args.decimal_points}f}), ({end_point_x:.{args.decimal_points}f}, {end_point_y:.{args.decimal_points}f}))"
            else:
                return f".moveTo({curr_x}, {curr_y}).threePointArc(({mid_point_x}, {mid_point_y}), ({end_point_x}, {end_point_y}))"
        else:
            if args.use_fixed_decimal is True:
                return f".threePointArc(({mid_point_x:.{args.decimal_points}f}, {mid_point_y:.{args.decimal_points}f}), ({end_point_x:.{args.decimal_points}f}, {end_point_y:.{args.decimal_points}f}))"
            else:
                return f".threePointArc(({mid_point_x}, {mid_point_y}), ({end_point_x}, {end_point_y}))"
        
    def cadquery_circle(x, y, r, loop_list, unquantize, extrude_scale):
        if unquantize:
            scale = extrude_scale / (256 / 2 * NORM_FACTOR - 1)
            translate = -(256 / 2)
            x = (x + translate)*scale
            y = (y + translate)*scale
            r = r*scale
        if len(loop_list) == 0:
            return f".moveTo({x:.{args.decimal_points}f}, {y:.{args.decimal_points}f}).circle({r:.{args.decimal_points}f})" if args.use_fixed_decimal else f".moveTo({x}, {y}).circle({r})"
        else:
            return NotImplementedError("Circle with other things in loop")
    
    def cadquery_close_loop(loop_operations, loop_num, sketch_num):
        if len(loop_operations) > 1:
            loop_string = "".join(loop_operations) + ".close()"
            return f"loop{loop_num}=wp_sketch{sketch_num}" + loop_string + "\n"
        else:
            if "circle" in loop_operations[0]:
                return f"loop{loop_num}=wp_sketch{sketch_num}" + loop_operations[0] + "\n"
            elif "Arc" in loop_operations[0]:
                loop_string = "".join(loop_operations) + ".close()"
                return f"loop{loop_num}=wp_sketch{sketch_num}" + loop_string + "\n"
            elif (len(loop_operations) == 1) and (loop_operations[0] == ''): # loop is single line going to 0,0 (effectively nothin)
                raise TypeError("Empty Loop")
            else:
                raise NotImplementedError(f"1 loop something else:{loop_operations}")
        
    def cadquery_extrude(loops_joined, op, type, dir1, dir2, sketch_num, normal, repeat_sketch, repeat_loops):
        
        # print(f"\tEXTRUDE INFO: {EXTRUDE_OPERATIONS[op], EXTENT_TYPE[type]}")
        
        # handle first body case
        if sketch_num == 0: #TODO: futher investigate case 80443. DeepCAD doesn't actually join the bodies? If the bodies are joined, shouldn't lines be removed from the pieces?
            if (op == EXTRUDE_OPERATIONS.index("NewBodyFeatureOperation")) or (op == EXTRUDE_OPERATIONS.index("JoinFeatureOperation")) or (op == EXTRUDE_OPERATIONS.index("CutFeatureOperation")) or (op == EXTRUDE_OPERATIONS.index("IntersectFeatureOperation")):
                if type == EXTENT_TYPE.index("OneSideFeatureExtentType"):
                    if args.use_fixed_decimal is True:
                        extrude_command = f".extrude({dir1:.{args.decimal_points}f})\n"
                    else:
                        extrude_command = f".extrude({dir1})\n"
                    return f"solid{sketch_num}=wp_sketch{sketch_num}" + loops_joined + extrude_command + f"solid=solid{sketch_num}\n"
                elif type == EXTENT_TYPE.index("SymmetricFeatureExtentType"):
                    if args.use_fixed_decimal is True:
                        extrude_command = f".extrude({dir1:.{args.decimal_points}f}, both=True)\n"
                    else:
                        extrude_command = f".extrude({dir1}, both=True)\n"
                    return f"solid{sketch_num}=wp_sketch{sketch_num}" + loops_joined + extrude_command + f"solid=solid{sketch_num}\n"
                else: # two sided extent
                    if (dir2==0): # I don't think DeepCAD handles this case
                        print("in this case")
                        return cadquery_extrude(loops_joined, op, EXTENT_TYPE.index("OneSideFeatureExtentType"), dir1, dir2, sketch_num, normal, repeat_sketch, repeat_loops)
                    
                    elif (dir1>0) and (dir2>0):
                        dir_with_normal = dir1
                        dir_against_normal = -dir2
    
                    elif (dir1<0) and (dir2>0): # I suspect this case was handled incorrectly by DeepCAD, and that if dir1 is negative it against normal while dir2 becomes direction of normal, matching their implementation
                        dir_with_normal = dir1
                        dir_against_normal = -dir2
                    
                    elif (dir1>0) and (dir2<0): # If the two bodies are completley overlapping, union gives an empty result. Example of this happening is index 5193. Happens in DeepCAD generation too
                        dir_with_normal = dir1
                        dir_against_normal = -dir2
                        print(f"dir_with: {dir_with_normal}, dir_against: {dir_against_normal}")
                        
                    elif (dir1<0) and (dir2<0): # copy from below case, not concretely tested
                        dir_with_normal = dir1
                        dir_against_normal = -dir2
                        
                    else:
                        raise NotImplementedError(f"Other dir cases this one: {dir1, dir2}")
                    
                    extrude_command_with_normal = f".extrude({dir_with_normal})\n" if not args.use_fixed_decimal else f".extrude({dir_with_normal:.{args.decimal_points}f})\n"
                    extrude_command_against_normal = f".extrude({dir_against_normal})\n" if not args.use_fixed_decimal else f".extrude({dir_against_normal:.{args.decimal_points}f})\n"
                    full_extrude =  f"solid{sketch_num}=wp_sketch{sketch_num}" + loops_joined + extrude_command_with_normal \
                        + f"solid=solid{sketch_num}\n" \
                        + repeat_sketch.split("\n")[1] + "\n" \
                        + repeat_loops \
                        + f"solid{sketch_num}=wp_sketch{sketch_num}" + loops_joined + extrude_command_against_normal \
                        + f"solid=solid.union(solid{sketch_num})\n"
                        
                    return full_extrude
                    # normal = normal/np.linalg.norm(normal) # get unit normal, if not already unit normal
                    # total_extrude = dir1 + dir2 # get total extrude as the sum of the two extrudes
                    # symmetric_extrude = total_extrude/2 # symmetric distance extrude is half of total distance
                    # extrude_command = f".extrude({symmetric_extrude}, both=True)\n"
                    # d = dir1 - symmetric_extrude
                    # translate_vector = d*normal # get how much we need to translate object to get nonsymmetric extrude
                    # return f"solid{sketch_num}=wp_sketch{sketch_num}" + loops_joined + extrude_command + f"solid=solid{sketch_num}\nsolid=solid.translate(({translate_vector[0]}, {translate_vector[1]}, {translate_vector[2]}))" #TODO: flip the order of the translate and assignment
            else:
                raise NotImplementedError("first body not newbody or join")
        
        # handle non first body cases
        else:
            if (op == EXTRUDE_OPERATIONS.index("NewBodyFeatureOperation")) or (op == EXTRUDE_OPERATIONS.index("JoinFeatureOperation")):
                if type == EXTENT_TYPE.index("OneSideFeatureExtentType"):
                    extrude_command = f".extrude({dir1})\n" if not args.use_fixed_decimal else f".extrude({dir1:.{args.decimal_points}f})\n"
                    return f"solid{sketch_num}=wp_sketch{sketch_num}" + loops_joined + extrude_command + f"solid=solid.union(solid{sketch_num})\n"
                elif type == EXTENT_TYPE.index("SymmetricFeatureExtentType"):
                    extrude_command = f".extrude({dir1}, both=True)\n" if not args.use_fixed_decimal else f".extrude({dir1:.{args.decimal_points}f}, both=True)\n"
                    return f"solid{sketch_num}=wp_sketch{sketch_num}" + loops_joined + extrude_command + f"solid=solid.union(solid{sketch_num})\n"
                else: # two sided extent
                    if (dir2==0): # should just be one-sided extrude anyway
                        return cadquery_extrude(loops_joined, op, EXTENT_TYPE.index("OneSideFeatureExtentType"), dir1, dir2, sketch_num, normal, repeat_sketch, repeat_loops)
                    
                    elif (dir1>0) and (dir2>0):
                        dir_with_normal = dir1
                        dir_against_normal = -dir2
    
                    elif (dir1<0) and (dir2>0):
                        dir_with_normal = dir1
                        dir_against_normal = -dir2
                    
                    elif (dir1<0) and (dir2<0):
                        dir_with_normal = dir1
                        dir_against_normal = -dir2
                        
                    elif (dir1>0) and (dir2<0): # I think this is true, may need more concrete example
                        dir_with_normal = dir1
                        dir_against_normal = -dir2
                    else:
                        raise NotImplementedError(f"Other dir cases here: {dir1}, {dir2}")
                    
                    extrude_command_with_normal = f".extrude({dir_with_normal})\n" if not args.use_fixed_decimal else f".extrude({dir_with_normal:.{args.decimal_points}f})\n"
                    extrude_command_against_normal = f".extrude({dir_against_normal})\n" if not args.use_fixed_decimal else f".extrude({dir_against_normal:.{args.decimal_points}f})\n"
                    full_extrude =  f"solid{sketch_num}=wp_sketch{sketch_num}" + loops_joined + extrude_command_with_normal \
                        + f"solid_temp=solid{sketch_num}\n" \
                        + repeat_sketch.split("\n")[1] + "\n" \
                        + repeat_loops \
                        + f"solid{sketch_num}=wp_sketch{sketch_num}" + loops_joined + extrude_command_against_normal \
                        + f"solid_temp=solid_temp.union(solid{sketch_num})\n" \
                        + f"solid=solid.union(solid_temp)\n"
                        
                    return full_extrude
                
            elif (op == EXTRUDE_OPERATIONS.index("CutFeatureOperation")):
                if type == EXTENT_TYPE.index("OneSideFeatureExtentType"):
                    extrude_command = f".extrude({dir1})\n" if not args.use_fixed_decimal else f".extrude({dir1:.{args.decimal_points}f})\n"
                    return f"solid{sketch_num}=wp_sketch{sketch_num}" + loops_joined + extrude_command + f"solid=solid.cut(solid{sketch_num})\n"
                elif EXTENT_TYPE.index("SymmetricFeatureExtentType"):
                    extrude_command = f".extrude({dir1}, both=True)\n" if not args.use_fixed_decimal else f".extrude({dir1:.{args.decimal_points}f}, both=True)\n"
                    return f"solid{sketch_num}=wp_sketch{sketch_num}" + loops_joined + extrude_command + f"solid=solid.cut(solid{sketch_num})\n"
                else:
                    raise NotImplementedError("cut other extent")
            else: # this is an intersect
                if type == EXTENT_TYPE.index("OneSideFeatureExtentType"):
                    extrude_command = f".extrude({dir1})\n" if not args.use_fixed_decimal else f".extrude({dir1:.{args.decimal_points}f})\n"
                    return f"solid{sketch_num}=wp_sketch{sketch_num}" + loops_joined + extrude_command + f"solid=solid.intersect(solid{sketch_num})\n"
                elif type == EXTENT_TYPE.index("SymmetricFeatureExtentType"):
                    extrude_command = f".extrude({dir1}, both=True)\n" if not args.use_fixed_decimal else f".extrude({dir1:.{args.decimal_points}f}, both=True)\n"
                    return f"solid{sketch_num}=wp_sketch{sketch_num}" + loops_joined + extrude_command + f"solid=solid.intersect(solid{sketch_num})\n"
                else: # two sided extent
                    if (dir1>0) and (dir2>0):
                        dir_with_normal = dir1
                        dir_against_normal = -dir2
    
                    elif (dir1<0) and (dir2>0):
                        dir_with_normal = dir1
                        dir_against_normal = -dir2
                    
                    else:
                        raise NotImplementedError("Other dir cases")
                    
                    extrude_command_with_normal = f".extrude({dir_with_normal})\n" if not args.use_fixed_decimal else f".extrude({dir_with_normal:.{args.decimal_points}f})\n"
                    extrude_command_against_normal = f".extrude({dir_against_normal})\n" if not args.use_fixed_decimal else f".extrude({dir_against_normal:.{args.decimal_points}f})\n"
                    full_extrude =  f"solid{sketch_num}=wp_sketch{sketch_num}" + loops_joined + extrude_command_with_normal \
                        + f"solid_temp=solid{sketch_num}\n" \
                        + repeat_sketch.split("\n")[1] + "\n" \
                        + repeat_loops \
                        + f"solid{sketch_num}=wp_sketch{sketch_num}" + loops_joined + extrude_command_against_normal \
                        + f"solid_temp=solid_temp.union(solid{sketch_num})\n" \
                        + f"solid=solid.intersect(solid_temp)\n"
                        
                    return full_extrude

    def join_loops(loops, sketch_num):
        loops_expanded = [f".add(loop{l})" for l in loops]
        return "".join(loops_expanded)
    
    # Initiate python program string
    python_cadquery = "import cadquery as cq\n"
    
    # print(f"Original numpy array:\n{vecs}")
    
    sketches, extrudes = split_by_sketches(vecs)
    
    # Initialize loop counter
    total_loop_count = 0
    curr_x = 128
    curr_y = 128
    
    for i, sketch in enumerate(sketches):
        # print(f"Sketch/extrude: {i}")
        extrude = extrudes[i] # get the corresponding extrude to the sketch
        theta, phi, gamma, px, py, pz, scale, dir1, dir2, op, type = extract_extrude_params(extrude, UNQUANTIZE) # extract individual extrude params
        
        if (dir1 == 0):
            raise TypeError("Extrude is zero for dir1 or for both directions") # DeepCAD doesn't handle cases where Dir1 is 0 but Dir2 has a value
        
        if (dir2 == 0) and (type == EXTENT_TYPE.index("TwoSidesFeatureExtentType")):
            raise TypeError("Two sided extrude with no second extrude value") # DeepCAD doesn't handle cases with two sided extrude and Dir2 is zero
        
        sketch_plane = CoordSystem(np.array([px, py, pz]), theta, phi, gamma) # define coordinate system
        sketch_plane.denumericalize(256) # denumericalize, unquantize
        # print(f"Sketch plane: {sketch_plane}")
        
        cadquery_sketch_plane = cadquery_workplane(sketch_plane, i)
        python_cadquery += cadquery_sketch_plane
        # print(f"\tCADQuery sketch plane: {cadquery_sketch_plane}")
        
        loops = split_by_loops(sketch)
        
        loops_in_this_sketch = []
        
        loop_chunk = ""
        
        for loop in loops:
            # print("\tLoop:")
            # print(f"\t{loop}")
            
            loop_ops = []
            
            for s_ind, sketch_op in enumerate(loop):
                # print(f"\tSketchOp:{sketch_op}")
                
                if sketch_op[0] == LINE:
                    # print("found line")
                    
                    point_x = sketch_op[1]
                    point_y = sketch_op[2]
                        
                    cadquery_line_command = cadquery_line(point_x, point_y, curr_x, curr_y, loop_ops, UNQUANTIZE, scale)
                    # python_cadquery += cadquery_line_command
                    
                    curr_x = point_x
                    curr_y = point_y
                    loop_ops.append(cadquery_line_command)
                    
                elif sketch_op[0] == ARC:
                    
                    end_x = sketch_op[1]
                    end_y = sketch_op[2]
                    sweep_angle = sketch_op[3]
                    directional_flag = sketch_op[4]
                    
                    cadquery_arc_command = cadquery_arc(end_x, end_y, curr_x, curr_y, sweep_angle, directional_flag, loop_ops, UNQUANTIZE, scale)
                
                        
                    curr_x = end_x
                    curr_y = end_y
                    loop_ops.append(cadquery_arc_command)
                
                elif sketch_op[0] == CIRCLE:
                    
                    center_x = sketch_op[1]
                    center_y = sketch_op[2]
                    circle_radius = sketch_op[5]
                    
                    cadquery_circle_command = cadquery_circle(center_x, center_y, circle_radius, loop_ops, UNQUANTIZE, scale)
                    
                    loop_ops.append(cadquery_circle_command)
                    
                else:
                    raise NotImplementedError("Command not implemented")
                
            # Close the loop
            cadquery_loop_closure_command = cadquery_close_loop(loop_ops, total_loop_count, i)
            python_cadquery += cadquery_loop_closure_command
            loop_chunk += cadquery_loop_closure_command # for feeding to extrude function for asymmetric extrusion
            
            # Track which loops are part of this sketch/extrude
            loops_in_this_sketch.append(total_loop_count)
            total_loop_count += 1
        
        # Handling the extrude below here
        # print(f"Loops in this sketch: {loops_in_this_sketch}")
        # print(f"Extrude type: {op}, {type}")
        # print(f"Extrude dimensions: {dir1, dir2}")
        loops_joined = join_loops(loops_in_this_sketch, i)
        
        cadquery_extrude_command = cadquery_extrude(loops_joined, op, type, dir1, dir2, i, sketch_plane.normal, cadquery_sketch_plane, loop_chunk)
        
        python_cadquery += cadquery_extrude_command
    
    # Write the python string to a python file
    prefix = '/'.join(H5_VEC_FOLDER.split('/', 2)[:2])
    
    #save_stl = f"{prefix}/cadquery_stl/" + save_python_dir[24:-3] + ".step" 
    # # TODO: get rid of hard coding of save root dir
    
    if generate_stls:
        python_cadquery += f"cq.exporters.export(solid, \"{save_step_path}\")"
    os.makedirs(save_python_dir.rsplit("/", 1)[0], exist_ok=True)
    with open(save_python_dir, "w") as file:
        file.write(python_cadquery)
    
def step_checker (save_python_dir):
        """
        Checks if the generated python file produces a valid STEP file
        """
        try:
            subprocess.run(["python", save_python_dir], check=True)
            print(f"Generated STEP file: {save_python_dir}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error generating STEP file: {e}")
            return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="h5 to cadquery conversion")
    parser.add_argument("--delete_existing", action="store_true", help="Delete existing data if present (default: False)")
    parser.add_argument("--use_fixed_decimal", action="store_true", help="Truncate decimal points (default: False)")
    parser.add_argument("--decimal_points", type=int, default=0, help="Number of decimal points to give floating point values. Can only use this with --use_fixed_decimal")
    args = parser.parse_args()
    
    if args.delete_existing:
        shutil.rmtree(os.path.dirname(H5_VEC_FOLDER) + "/cadquery_stl/")
        shutil.rmtree(os.path.dirname(H5_VEC_FOLDER) + "/cadquery/")
    # Set up save cadquery save directory
    root_save_dir = create_save_dir(H5_VEC_FOLDER)

   
#file_path_ques = str(input("What file path would you like to save the generated python files to? (default: deepcad_derived/data/cad_vec/cadquery): "))

    prefix = '/'.join(H5_VEC_FOLDER.split('/', 2)[:2])
    
    # TODO: iterate through all of the h5 files at this point
    # h5_vec_path = h5_files[35167]
    # h5_vec_path = h5_files[95149]
    
    #sub_dirs = [str(i).zfill(4) for i in range(100)]
    sub_dirs = ["0000"]

    for sub_dir in sub_dirs:
        
        code_files_not_generated = 0 # count how many code files are not generated
        code_files_successfully_generated = 0
        stls_generated_successfully = 0
        stls_not_generated = 0
        gen_file = 0 # count how many files were generated
        no_code = []
        no_stl = []
        file_difference_from_deepcad = []
        
        print(f"Processing Subdir: {sub_dir}")
        
        # Get list of all .h5 files in subdirectories
        h5_files = glob.glob(f"{H5_VEC_FOLDER}/{sub_dir}/*.h5")
        
        if generate_stls:
            step_dir_root = os.path.dirname(H5_VEC_FOLDER) + "/cadquery_stl/" + sub_dir
            os.makedirs(step_dir_root, exist_ok=True)  # make subfolder if it doesn't exist already
        
        # If you want a specific h5 file, collapse the loop under this
        # h5_vec_path = h5_files[690]
        # print(h5_vec_path)
        
        for i, h5_vec_path in enumerate(h5_files):
            
            print(f"Num: {i}")
            print(f"Path: {h5_vec_path}")
            
            python_file_save_path = root_save_dir + h5_vec_path.replace(H5_VEC_FOLDER, "").rsplit(".", 1)[0] + ".py"
            stl_file_save_path = f"{H5_VEC_FOLDER[:-5]}/cadquery_stl/" + h5_vec_path.replace(H5_VEC_FOLDER, "").rsplit(".", 1)[0] + ".stl"
            unique_id = python_file_save_path.split(".")[0].removeprefix(os.path.dirname(H5_VEC_FOLDER) + "/cadquery")
            step_file_save_path = os.path.dirname(H5_VEC_FOLDER) + "/cadquery_stl/" + unique_id + ".step"
            
            h5_vec = extract_h5_file(h5_vec_path)
            
            try:
                convert_h5_to_cadquery(h5_vec, python_file_save_path, step_file_save_path, args.use_fixed_decimal, args.decimal_points)
                code_files_successfully_generated += 1
                
                if generate_stls:
                    try:
                        
                        if ("/0002/00024147.h5" not in h5_vec_path) and ("/0005/00057125.h5" not in h5_vec_path) and ("/0012/00126522.h5" not in h5_vec_path) and ("/0014/00140564.h5" not in h5_vec_path): #handle these weird hanging cases
                            subprocess.run(["python", python_file_save_path], check=True)
                            stls_generated_successfully += 1
                            if step_checker(python_file_save_path): # check if the generated python file produces a valid STEP file
                                gen_file += 1

                            # TODO: Check for equivalence of STLs, create a log of stl differences
                            
                        else:
                            stls_not_generated += 1
                            no_stl.append(h5_vec_path.replace(H5_VEC_FOLDER, "").rsplit(".", 1)[0])
                            
                            # Check if DeepCAD produced the file
                            if os.path.exists(f"{prefix}/deepcad_stl" + h5_vec_path.replace(H5_VEC_FOLDER, "").rsplit(".", 1)[0] + ".stl"):
                                file_difference_from_deepcad.append("hanging case: " + h5_vec_path.replace(H5_VEC_FOLDER, "").rsplit(".", 1)[0])
  
                    except subprocess.CalledProcessError as e:
                        print("Error occurred:", e)
                        stls_not_generated += 1
                        no_stl.append(h5_vec_path.replace(H5_VEC_FOLDER, "").rsplit(".", 1)[0])
                        
                        # Check if DeepCAD produced the file
                        if os.path.exists(f"{prefix}/deepcad_stl" + h5_vec_path.replace(H5_VEC_FOLDER, "").rsplit(".", 1)[0] + ".stl"):
                            file_difference_from_deepcad.append("stl: " + h5_vec_path.replace(H5_VEC_FOLDER, "").rsplit(".", 1)[0])
                        
            except TypeError as e:
                print(f"ERROR: {h5_vec_path} NO GENERATION: {e}")
                code_files_not_generated += 1
                no_code.append(h5_vec_path.replace(H5_VEC_FOLDER, "").rsplit(".", 1)[0])
                
                # Check if DeepCAD produced the file
                if os.path.exists(f"{prefix}/deepcad_stl" + h5_vec_path.replace(H5_VEC_FOLDER, "").rsplit(".", 1)[0] + ".stl"):
                    file_difference_from_deepcad.append("code: " + h5_vec_path.replace(H5_VEC_FOLDER, "").rsplit(".", 1)[0])
    
        print(f"Total number of h5_vec files: {len(h5_files)}")
        print(f"Successful code file generations: {code_files_successfully_generated}")
        print(f"Code file fails: {code_files_not_generated}")
        print(f"STLs generated successfully: {stls_generated_successfully}")
        print(f"STLs NOT generated: {stls_not_generated}")
        print (f"Total number of generated files: {gen_file}")
    
    
        with open(f"{prefix}/logs/code_issues_" + sub_dir + ".txt", "w") as f:
            f.write("\n".join(no_code))
        
        with open(f"{prefix}/logs/stl_issues" + sub_dir + ".txt", "w") as f:
            f.write("\n".join(no_stl))
        
        with open(f"{prefix}/logs/log_files_" + sub_dir + ".txt", "w") as f:
            f.write("\n".join(file_difference_from_deepcad))

        # Log the truncation level and the accuracy it brings
        trunc_path = f"{prefix}/trunc_logs.txt"
        if os.path.exists(trunc_path):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not
        with open (f"{prefix}/trunc_logs.txt", append_write) as f:
            f.write(f"{args.decimal_points}, {gen_file}\n")
        
    
    # for i, h5_vec_path in enumerate(h5_files):
        
    #     if '/0000/' in h5_vec_path:
        
    #         if "00004255.h5" in h5_vec_path:
                
    #             print(f"Num: {i}")
            
    #         python_file_save_path = root_save_dir + h5_vec_path.replace(H5_VEC_FOLDER, "").rsplit(".", 1)[0] + ".py"
            
    #         # print(f"Test file: {h5_vec_path}")
    #         # print(f"Test save: {python_file_save_path}")
            
    #         h5_vec = extract_h5_file(h5_vec_path)
            
    #         try:
    #             convert_h5_to_cadquery(h5_vec, python_file_save_path)
    #         except TypeError as e:
    #             print(f"ERROR: {h5_vec_path} NO GENERATION: {e}")
    #             code_files_not_generated += 1

    # print(f"Didn't generate: {code_files_not_generated}")