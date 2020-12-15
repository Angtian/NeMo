kp_list_dict = {'aeroplane': ['left_wing', 'right_wing', 'rudder_upper', 'noselanding', 'left_elevator', 'rudder_lower', 'right_elevator', 'tail'], 
                'bicycle': ['seat_front', 'right_back_wheel', 'right_pedal_center', 'right_front_wheel', 'left_front_wheel', 'left_handle', 'seat_back', 'head_center', 'left_back_wheel', 'left_pedal_center', 'right_handle'], 
                'boat': ['head_down', 'head', 'tail_right', 'tail_left', 'head_right', 'tail', 'head_left'], 
                'bottle': ['body', 'bottom_left', 'bottom', 'mouth', 'body_right', 'body_left', 'bottom_right'], 
                'bus': ['body_front_left_lower', 'body_front_right_upper', 'body_back_right_lower', 'right_back_wheel', 'body_back_left_upper', 'right_front_wheel', 'left_front_wheel', 'body_front_left_upper', 'body_back_left_lower', 'body_back_right_upper', 'body_front_right_lower', 'left_back_wheel'], 
                'car': ['left_front_wheel', 'left_back_wheel', 'right_front_wheel', 'right_back_wheel', 'upper_left_windshield', 'upper_right_windshield', 'upper_left_rearwindow', 'upper_right_rearwindow', 'left_front_light', 'right_front_light', 'left_back_trunk', 'right_back_trunk'], 
                'chair': ['seat_upper_right', 'back_upper_left', 'seat_lower_right', 'leg_lower_left', 'back_upper_right', 'leg_upper_right', 'seat_lower_left', 'leg_upper_left', 'seat_upper_left', 'leg_lower_right'], 
                'diningtable': ['top_lower_left', 'top_up', 'top_lower_right', 'leg_lower_left', 'leg_upper_right', 'top_right', 'top_left', 'leg_upper_left', 'top_upper_left', 'top_upper_right', 'top_down', 'leg_lower_right'], 
                'motorbike': ['front_seat', 'right_back_wheel', 'back_seat', 'right_front_wheel', 'left_front_wheel', 'headlight_center', 'right_handle_center', 'left_handle_center', 'head_center', 'left_back_wheel'], 
                'sofa': ['top_right_corner', 'seat_bottom_right', 'left_bottom_back', 'seat_bottom_left', 'front_bottom_right', 'top_left_corner', 'right_bottom_back', 'seat_top_left', 'front_bottom_left', 'seat_top_right'], 
                'train': ['head_top', 'mid1_left_bottom', 'head_left_top', 'mid1_left_top', 'mid2_right_bottom', 'head_right_bottom', 'mid1_right_bottom', 'head_left_bottom', 'mid2_left_top', 'mid2_left_bottom', 'head_right_top', 'tail_right_top', 'tail_left_top', 'tail_right_bottom', 'tail_left_bottom', 'mid2_right_top', 'mid1_right_top'], 
                'tvmonitor': ['back_top_left', 'back_bottom_right', 'front_bottom_right', 'front_top_right', 'front_top_left', 'back_bottom_left', 'back_top_right', 'front_bottom_left']}

mesh_len = {'aeroplane': 8, 'bicycle': 6, 'boat': 6, 'bottle': 8, 'bus': 6, 'car': 10, 'chair': 10, 'diningtable': 6, 'motorbike': 5, 'sofa': 6, 'train': 4, 'tvmonitor': 4}

max_size_dict = {'car' : [256., 672.],
'aeroplane' : [ 320., 1024.],
'bicycle' : [600., 600.],
'boat' : [ 512., 1200.],
'bottle' : [512., 720.],
'bus' : [382., 896.],
'chair' : [256., 320.],
'diningtable' : [320., 800.],
'motorbike' : [512., 512.],
'sofa' : [337., 720.],
'train' : [ 450., 1000.],
'tvmonitor' : [468., 463.],
}

# For NeMo
top_50_size_dict = {
'car' : [256.0, 672.0] ,
'aeroplane' : [320., 1024.],
'bicycle' : [600.0, 600.0] ,
'boat' : [455.0, 1120.0] ,
'bottle' : [512., 720.] ,
'bus' : [320.0, 800.0] ,
'chair' : [530., 370.] ,
'diningtable' : [320., 800.],
'motorbike' : [512., 512.],
'sofa' : [337., 720.] ,
'train' : [240., 600.] ,
'tvmonitor' : [468., 463.] ,
}
