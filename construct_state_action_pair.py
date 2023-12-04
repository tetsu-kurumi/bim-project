import json
import yaml
import sys

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

def extract_head_positions(yaml_text_file):
    head_position_dict = {}
    with open (yaml_text_file, 'r') as yaml_text:
        data_list = list(yaml.safe_load_all(yaml_text))
        for data in data_list:
            if data is not None:
                head_data = data['markers'][16]
                secs = head_data['header']['stamp']['secs']
                x = head_data['pose']['position']['x']
                y = head_data['pose']['position']['y']
                z = head_data['pose']['position']['z']
                transformed_x, transformed_y, transformed_z = transform(x, y, z)
                # x_orientation = ['pose']['orientation']['x']
                # y_orientation = ['pose']['orientation']['x']
                # z_orientation = ['pose']['orientation']['x']
                head_position_dict[secs] = (transformed_x, transformed_y, transformed_z)
    return head_position_dict

def transform(x, y, z):
   return x, y, z

def construct_state_action_pair(joint_positions_yaml, head_positions_yaml, output_json):
    # Parse the YAML text
    joint_position_dict = extract_joint_positions(joint_positions_yaml)
    head_position_dict = extract_head_positions(head_positions_yaml)

    for secs, joint_position in joint_position_dict.items():
        if secs in head_position_dict:
            head_position = head_position_dict[secs]
            if int(secs)+1 in joint_position_dict:
                next_joint_position = joint_position_dict[secs+1]
                with open(output_json, 'a') as json_file:
                    data_to_write = {"secs": secs, "head_position_x": head_position[0], "head_position_y": head_position[1], "head_position_z": head_position[2],
                                    "joint_1": joint_position[0], "joint_2": joint_position[1], "joint_3": joint_position[2], "joint_4": joint_position[3], 
                                    "next_joint_1": next_joint_position[0], "next_joint_2": next_joint_position[1], "next_joint_3": next_joint_position[2], "next_joint_4": next_joint_position[3]}
                    json_file.seek(0, 2)
                    json_file.write(json.dumps(data_to_write) + '\n')


if __name__ == "__main__":
    joint_positions_yaml  = sys.argv[1]
    head_positions_yaml = sys.argv[2]
    output_json = sys.argv[3]
    construct_state_action_pair(joint_positions_yaml, head_positions_yaml, output_json)
    print(f"Conversion completed. Data written to {output_json}")