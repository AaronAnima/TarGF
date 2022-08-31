from shutil import copy
import os
import igibson

ig_path = igibson.root_path

copy('./Misc/scene_loader.py', ig_path)
copy('./Misc/floor_map_utils.py', f'{ig_path}/scenes/')
copy('./Misc/igibson_indoor_scene.py',  f'{ig_path}/scenes/')
copy('./Misc/articulated_object.py', f'{ig_path}/objects/')
copy('./Misc/assets_utils.py', f'{ig_path}/utils/')

# create soft link
src = './data'
dst = f'{igibson.root_path}/data'
os.symlink(src, dst)