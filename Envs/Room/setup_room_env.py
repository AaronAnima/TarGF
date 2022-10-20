from shutil import copy
import os
import igibson

ig_path = igibson.root_path

copy('./CustomModules/scene_loader.py', ig_path)
copy('./CustomModules/floor_map_utils.py', f'{ig_path}/scenes/')
copy('./CustomModules/igibson_indoor_scene.py',  f'{ig_path}/scenes/')
copy('./CustomModules/articulated_object.py', f'{ig_path}/objects/')
copy('./CustomModules/assets_utils.py', f'{ig_path}/utils/')

# create soft link
# src -> dst
src = os.path.join(os.getcwd(), './data')
dst = f'{igibson.root_path}/data'
os.symlink(src, dst)
