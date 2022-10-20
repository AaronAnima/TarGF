bedroom_category_num_dict = {'bed': 3880, 'bottom_cabinet': 9744, 'coffee_table': 568,
                             'chair': 919, 'shelf': 674, 'table': 1031,
                             'stool': 294, 'sofa_chair': 181, 'sofa': 117, 'top_cabinet': 1}

living_room_category_num_dict = {'sofa': 1410, 'coffee_table': 2178, 'bottom_cabinet': 1379,
                             'sofa_chair': 742, 'shelf': 869, 'stool': 340,
                             'chair': 2407, 'table': 697, 'bed': 28}

bedroom_category_subtype_dict = {'bed': 12, 'bottom_cabinet': 14, 'coffee_table': 15, 'chair': 18, 'shelf': 18,
                         'table': 15, 'stool': 12, 'sofa_chair': 11, 'sofa': 12, 'top_cabinet': 1}

bedroom_num_per_room_dict = {0: 373, 3: 952, 4: 1168, 5: 696, 2: 504,
                            6: 366, 7: 195, 8: 89, 1: 202, 11: 9,
                            9: 43, 10: 25, 12: 5, 19: 1, 14: 4, 13: 1, 18: 1, 16: 1}

bedroom_type_mapping = {key: idx for idx, key in enumerate(bedroom_category_num_dict.keys())}
bedroom_typeidx_mapping = {idx: key for idx, key in enumerate(bedroom_category_num_dict.keys())}
livingroom_type_mapping = {key: idx for idx, key in enumerate(living_room_category_num_dict.keys())}
livingroom_typeidx_mapping = {idx: key for idx, key in enumerate(living_room_category_num_dict.keys())}


bedroom_category_num_dict = {'bed': 3880, 'bottom_cabinet': 9744, 'coffee_table': 568,
                             'chair': 919, 'shelf': 674, 'table': 1031,
                             'stool': 294, 'sofa_chair': 181, 'sofa': 117, 'top_cabinet': 1}

#