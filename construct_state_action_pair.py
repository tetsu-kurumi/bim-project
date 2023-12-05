import json
import yaml
import sys
import numpy as np

def extract_joint_positions(yaml_text_file):
    joint_position_dict = {}
    with open (yaml_text_file, 'r') as yaml_text:
        data_list = list(yaml.safe_load_all(yaml_text))

        for index, data in enumerate(data_list):
            if data is not None:
                # read data
                secs = data['header']['stamp']['secs']
                joint_1 = data['position'][0]
                joint_2 = data['position'][1]
                joint_3 = data['position'][2]
                joint_4 = data['position'][3]

                # start adding data to the dictionary if the demonstration is happening (if the joint is moving)
                if index > 0:
                    if joint_1 != data_list[index - 1]['position'][0] or joint_2 != data_list[index - 1]['position'][1] or joint_3 != data_list[index - 1]['position'][2] or joint_4 != data_list[index - 1]['position'][3]:
                        joint_position_dict[secs] = (joint_1, joint_2, joint_3, joint_4)

    return joint_position_dict

def extract_head_positions(yaml_text_file, transformation_matrix):
    head_position_dict = {}
    with open (yaml_text_file, 'r') as yaml_text:
        data_list = list(yaml.safe_load_all(yaml_text))
        for data in data_list:
            if data is not None:
                if len(data['markers']) >= 17:
                    head_data = data['markers'][16]
                    secs = head_data['header']['stamp']['secs']
                    x = head_data['pose']['position']['x']
                    y = head_data['pose']['position']['y']
                    z = head_data['pose']['position']['z']
                    transformed_x, transformed_y, transformed_z = transform_3d_vector(np.array([x, y, z]), transformation_matrix)
                    # x_orientation = ['pose']['orientation']['x']
                    # y_orientation = ['pose']['orientation']['x']
                    # z_orientation = ['pose']['orientation']['x']
                    head_position_dict[secs] = (transformed_x, transformed_y, transformed_z)
    return head_position_dict

def transform_3d_vector(vector, transformation_matrix):
    vector_homogeneous = np.array([vector[0], vector[1], vector[2], 1.0])  # Homogeneous coordinates
    result_homogeneous = np.dot(transformation_matrix, vector_homogeneous)
    
    # Convert back to Cartesian coordinates
    result = result_homogeneous[:-1] / result_homogeneous[-1]
    
    return result

def construct_state_action_pair(joint_positions_yaml, head_positions_yaml, output_json, transformation_matrix):
    # Parse the YAML text
    joint_position_dict = extract_joint_positions(joint_positions_yaml)
    head_position_dict = extract_head_positions(head_positions_yaml, transformation_matrix)

    with open(output_json, 'a') as json_file:
        json_file.seek(0, 2)
        for secs, joint_position in joint_position_dict.items():
            if secs in head_position_dict:
                head_position = head_position_dict[secs]
                if int(secs)+1 in joint_position_dict:
                    next_joint_position = joint_position_dict[secs+1]
                    data_to_write = {"secs": secs, "head_position_x": head_position[0], "head_position_y": head_position[1], "head_position_z": head_position[2],
                                    "joint_1": joint_position[0], "joint_2": joint_position[1], "joint_3": joint_position[2], "joint_4": joint_position[3], 
                                    "next_joint_1": next_joint_position[0], "next_joint_2": next_joint_position[1], "next_joint_3": next_joint_position[2], "next_joint_4": next_joint_position[3]}
                    json_file.write(json.dumps(data_to_write) + '\n')

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02, 0],
                           [r10, r11, r12, 0],
                           [r20, r21, r22, 0],
                           [0, 0, 0, 1]])
                            
    return rot_matrix

if __name__ == "__main__":
    t = np.array([[1, 0, 0, -0.204],
                  [0, 1, 0, -0.032],
                  [0, 0, 1, 0.545],
                  [0, 0, 0, 1]])
    Q = (-0.499, 0.500, -0.499, 0.502)
    r = quaternion_rotation_matrix(Q)
    transformation_matrix = np.dot(t, r)
    joint_positions_yaml  = sys.argv[1]
    head_positions_yaml = sys.argv[2]
    output_json = sys.argv[3]
    construct_state_action_pair(joint_positions_yaml, head_positions_yaml, output_json, transformation_matrix)
    print(f"Conversion completed. Data written to {output_json}")