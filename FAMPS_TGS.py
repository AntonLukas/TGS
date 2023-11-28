__author__ = 'Anton Lukas'
########################################################################################################################
############## TGS Template Generation Script: Isotropic Hydraulic Conductivity and Storativity Edition ################
########################################################################################################################

import numpy as np
import pandas as pd
import random as rd
import os


def model_generator(layer_iterations: int, parameter_iterations: int) -> pd.DataFrame:
    # Define out dictionary
    models_dict = {}
    
    # For each layer iteration...
    for i in range(layer_iterations):
        # ...create a layer model
        model_layers = layer_configuration(np.random.rand())
        model_key = ""

        # Get the model configuration
        for key in model_layers.keys():
            model_key = str(key)
        
        # If a model with the same configuration does not exist, add it to the out dictionary
        if model_key not in models_dict.keys():
            models_dict[model_key] = model_layers[model_key]

    # Create DataFrame from model layer information
    df_models = pd.DataFrame(models_dict.items(), columns=["Model", "LayerConfig"])
    df_models["KValues"] = ""
    df_models["SValues"] = ""
    df_models["DFeatures"] = ""
    df_models.set_index('Model')

    # For each model assign multiple parameter distributions
    df_models = layer_parameterization(df_models, parameter_iterations)
    # Return the DataFrame
    return df_models


def layer_configuration(random_state) -> dict:
    # Maximum number of layers
    max_layers = 10
    # Assign the amount of layers left to use
    layers_left = max_layers
    # Define the output array and string
    model_dict = {}
    layer_array = []

    # While there are still layers left to assign
    while layers_left > 0:
        if layers_left > 1:
            # Assign a random value between 1 and the amount of layers left to create a hydraulic layer
            if random_state > 0.5 and layers_left % 2 == 0 and layers_left > 2:
                random_layer = np.random.randint(low=1, high=layers_left/2)
            else:
                random_layer = np.random.randint(low=1, high=layers_left)
        else:
            # If there is only 1 layer left then return 1
            random_layer = 1

        # Minus the amount of layers used from the amount of layers left
        layers_left = layers_left - random_layer
        # Add the layer to the output array
        layer_array.append(random_layer)

    # Shuffle the array
    rd.shuffle(layer_array)
    # Create a string describing the layer configuration
    model_string = ""
    # Assign the layer configuration string
    for value in layer_array:
        model_string = model_string + str(value)
    # Assign the output
    model_dict[model_string] = layer_array
    # Return the output
    return model_dict


def layer_parameterization(df_models: pd.DataFrame, parameter_iterations: int) -> pd.DataFrame:
    # For each model assign multiple parameter distributions
    for i in range(df_models.shape[0]):
        # Create dictionary to host the different parameter iterations
        k_params = "{"
        s_params = "{"
        # For each parameter iteration
        for j in range(parameter_iterations):
            # Prepare in array
            layer_array = df_models.LayerConfig.iloc[i]
            number_layers = len(layer_array)
            # Prepare output arrays
            k_params_out = []
            s_params_out = []
            # Keep track of sum of model hydraulic conductivities
            sum_of_k_values = 0
            # Create the arrays that serves as the base random
            k_random_array = np.random.rand(1, number_layers)
            s_random_array = np.random.rand(1, number_layers)

            # For each value in the base random array
            for icount in range(number_layers):
                # Get a second random value
                rvalue = np.random.rand()

                # Change the base random array value by means of classifying the second random value
                if rvalue > 0.8:
                    k_export_value = k_random_array[0, icount] * 1.0
                    s_export_value = s_random_array[0, icount] * 0.001
                elif rvalue > 0.6:
                    k_export_value = k_random_array[0, icount] * 0.5

                    if ((rvalue - 0.6) * 10) > 1:
                        s_export_value = s_random_array[0, icount] * 0.001
                    else:
                        s_export_value = s_random_array[0, icount] * 0.0005
                elif rvalue > 0.4:
                    k_export_value = k_random_array[0, icount] * 0.1
                    
                    if ((rvalue - 0.4) * 10) > 1:
                        s_export_value = s_random_array[0, icount] * 0.001
                    else:
                        s_export_value = s_random_array[0, icount] * 0.0005
                elif rvalue > 0.2:
                    k_export_value = k_random_array[0, icount] * 0.01

                    if ((rvalue - 0.2) * 10) > 1.2:
                        s_export_value = s_random_array[0, icount] * 0.001
                    elif ((rvalue - 0.2) * 10) > 0.60:
                        s_export_value = s_random_array[0, icount] * 0.0005
                    else:
                        s_export_value = s_random_array[0, icount] * 0.0001
                else:
                    k_export_value = k_random_array[0, icount] * 0.005
                    
                    if (rvalue * 10) > 1.2:
                        s_export_value = s_random_array[0, icount] * 0.001
                    elif (rvalue * 10) > 0.60:
                        s_export_value = s_random_array[0, icount] * 0.0005
                    else:
                        s_export_value = s_random_array[0, icount] * 0.0001

                # Add the new value to the output array
                k_params_out.append(k_export_value)
                s_params_out.append(s_export_value)
                sum_of_k_values = sum_of_k_values + k_export_value

            # Check that the values aren't unreasonably low
            average_of_values = sum_of_k_values / k_random_array.size

            while average_of_values < 0.001:
                print("Average: " + str(average_of_values))
                print("Layer Configuration: " + str(layer_array))
                print("Parameter Configuration (Old): " + str(k_params_out))

                random_layer = np.random.randint(1, k_random_array.size)
                k_params_out[random_layer] = k_params_out[random_layer] * 1000

                print("Parameter Configuration (New): " + str(k_params_out))

                average_of_values = sum(k_params_out) / len(k_params_out)

            # Create a parameter model and assign it to the dictionary string
            if j == (parameter_iterations - 1):
                k_params = k_params + "'" + str(j) + "': " + str(k_params_out) + "}"
                s_params = s_params + "'" + str(j) + "': " + str(s_params_out) + "}"
            else:
                k_params = k_params + "'" + str(j) + "': " + str(k_params_out) + ", "
                s_params = s_params + "'" + str(j) + "': " + str(s_params_out) + ", "
        
        # Generate the discrete feature elements
        d_features = dfe_generation()

        # Write the dictionary into the DataFrame
        df_models.KValues.iloc[i] = k_params
        df_models.SValues.iloc[i] = s_params
        df_models.DFeatures.iloc[i] = d_features

    return df_models


def dfe_generation(total_features: int = 3, total_feature_iterations: int = 8) -> str:
    # Feature information - {key: distance_from_pumping_borehole_class} <- determines order of feature iterations
    # Hardcoded for FAMPS base models.
    vertical_features = {
        1: 2, 2: 5, 3: 9, 4: 8, 5: 1, 6: 4, 7: 10, 8: 6, 9: 3, 10: 7
    }
    horizontal_features = {
        1: [17, 24, 31], 2: [16, 23, 30], 3: [15, 22, 29], 4: [14, 21, 28],
        5: [13, 20, 27], 6: [12, 19, 26], 7: [11, 18, 25]
    }
    # Prepare output dictionary
    d_features = "{"
    # Prepare output array
    d_features_out = []
    # Create index tracker for the keys
    key_tracker = []
    # Get which types of features are to be created
    feature_types = np.random.randint(0, 2, total_features)
    # Sort the feature types
    feature_types = np.sort(feature_types)
    # Create the discrete feature information
    for k in range(total_feature_iterations):
        # Create base variables
        feature_type = 0  # 0: Horizontal, 1: Vertical, 2: Arbitrary <- Removed for now
        feature_law = 0  # 0: Darcy, 1: HP, 2: MS
        feature_aperture = 0  # Aperture size in mm
        feature_index = 0  # Index of the feature in the model

        # For the empty model...
        if k == 0:
            # Append an empty string
            d_features_out.append("")

        # For the individual features...
        elif k <= total_features:
            # Generate a random feature type
            feature_type = feature_types[k - 1]
            # If the feature type is horizontal
            if feature_type == 0:
                # Then the law can only be HP, and an aperture is assigned
                feature_law = 1
                feature_aperture = np.random.randint(1, 21)
                # Get the lowest and highest available feature index
                feature_low = int(min(horizontal_features.keys()))
                feature_high = int(max(horizontal_features.keys()))
                # Assign the feature index based on relative distance from the pumping borehole
                random_index = np.random.randint(feature_low + 1, feature_high - (total_features - k))
                # Check if the index is the first one
                if random_index == 1:
                    selected_index = 1
                    selected_layer = horizontal_features.get(selected_index)
                    feature_index = selected_layer[np.random.randint(0, 3)]
                else:
                    selected_index = np.random.randint(feature_low, random_index)
                    selected_layer = horizontal_features.get(selected_index)
                    feature_index = selected_layer[np.random.randint(0, 3)]
                # Save the index
                key_tracker.append(feature_index)
                # Remove the chosen feature and nearer features from the options
                horiz_feat_keys = list(horizontal_features.keys())
                for key in horiz_feat_keys:
                    if key <= selected_index:
                        del horizontal_features[key]
            # If the feature type is vertical or arbitrary
            else:
                # Then the law can be either Darcy or HP
                feature_law = np.random.randint(0, 2)
                # If the law is Darcy...
                if feature_law == 0:
                    # Then we create an impermeable feature by setting the hydraulic conductivity value to zero
                    feature_aperture = 0
                    # If the law is HP however... 
                elif feature_law == 1:
                    # Then assign an aperture
                    feature_aperture = np.random.randint(1, 21)
                # Get the lowest and highest available feature index
                feature_low = int(min(vertical_features.keys()))
                feature_high = int(max(vertical_features.keys()))
                # Assign the feature index based on relative distance from the pumping borehole
                random_index = np.random.randint(feature_low + 1, feature_high - (total_features - k))
                # Check if the index is the first one
                if random_index == 1:
                    selected_index = 1
                    feature_index = vertical_features.get(selected_index)
                else:
                    selected_index = np.random.randint(feature_low, random_index)
                    feature_index = vertical_features.get(selected_index)
                # Save the index
                key_tracker.append(feature_index)
                # Get a list of all the remaining vertical features
                vert_feat_keys = list(vertical_features.keys())
                # Remove the chosen feature and nearer features from the options
                for key in vert_feat_keys:
                    if key <= selected_index:
                        del vertical_features[key]
                # Remove specific features based on the selected feature
                if selected_index == 6:
                    del vertical_features[9]
                elif feature_index == 2:
                    del vertical_features[8]
                elif feature_index == 4:  # FAMPS index 8
                    del vertical_features[10]  # FAMPS index 7
                    # Check if certain geometries occur
                    if 1 in key_tracker:
                        del vertical_features[9]  # FAMPS index 3
                elif feature_index == 3:  # FAMPS index 9
                    del vertical_features[7]  # FAMPS index 10
                    # Check if certain geometries occur
                    if 2 in key_tracker:
                        del vertical_features[8]  # FAMPS index 6

            # Create and append only a single feature
            d_features_out.append(f"\"{feature_type}a{feature_law}a{feature_aperture}a{feature_index}\"")

        # For the feature combinations...
        elif total_features < k < (total_feature_iterations - 1):
            # Define output string
            out_str = ""
            # Get current DFE combination number
            iter_count_number = int(k - total_features)
            # Then combine previous features into sets.
            if iter_count_number == 1:
                out_str = f"{d_features_out[1]}, {d_features_out[2]}"
            elif iter_count_number == 2:
                out_str = f"{d_features_out[1]}, {d_features_out[3]}"
            elif iter_count_number == 3:
                out_str = f"{d_features_out[2]}, {d_features_out[3]}"
            # Which is then appended to the output list
            d_features_out.append(out_str)

        # For the combination of all features
        elif k == (total_feature_iterations - 1):
            # Define output string
            out_str = f"{d_features_out[1]}, {d_features_out[2]}, {d_features_out[3]}"
            # Which is then appended to the output list
            d_features_out.append(out_str)
    
    # Create a discrete feature summary and assign it to the dictionary string
    for m, val in enumerate(d_features_out):
        if m == (len(d_features_out) - 1):
            d_features = f"{d_features}'{str(m)}': [{val}]" + "}"
        else:
            d_features = f"{d_features}'{str(m)}': [{val}], "
    
    return d_features


def main():
    # Name for file
    out_filename = "FAMPSTemplate"
    # Number of layer models
    layer_iterations = 30
    # Number of parameter models per layer model
    parameter_iterations = 10
    # Number of virtual machines to split the work across
    number_virtual_machines = 1
    # Set the output directory
    output_directory = os.path.join(os.getcwd(), "data")
    # Make sure the output directory exists
    if not os.path.exists(f"{output_directory}"):
        # If not, create it
        os.mkdir(f"{output_directory}")
    # Perform model generation
    df_models = model_generator(layer_iterations, parameter_iterations)

    # If number of machines is more than 1
    if number_virtual_machines > 1:
        # Get the number of models
        number_models = df_models.shape[0]
        # Export the generated models divided between each virtual machine
        for VM in range(number_virtual_machines):
            # Create the output range
            i_start = int(number_models / number_virtual_machines) * VM
            i_end = int(number_models / number_virtual_machines) * (VM + 1)
            # Create a new dataframe for each VM
            df_models_out = df_models[i_start:i_end].reset_index(drop=True)
            # Create the complete filename with extension
            complete_filename = f"{out_filename}_{VM+1}.xlsx"
            # Write the dataframe to a Excel file
            df_models_out.to_excel(os.path.join(output_directory, complete_filename))
    else:
        # Create the complete filename with extension
        complete_filename = f"{out_filename}.xlsx"
        # Write the dataframe to a Excel file
        df_models.to_excel(os.path.join(output_directory, complete_filename))


# This is a script that is meant to be run, not imported    
if __name__ == '__main__':
    main()
