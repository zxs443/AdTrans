import pandas as pd
import numpy as np
import os

def summarize_deviant_samples(predictions_file, data_stats_file, output_file, train_ratio=0.05, other_ratio=0.1):
    """Export top-|error| samples per split to CSV."""
    try:
        predictions_df = pd.read_csv(predictions_file)
    except FileNotFoundError:
        print(f"Error: Prediction file '{predictions_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading prediction file: {e}")
        return

    try:
        data_stats = np.load(data_stats_file, allow_pickle=True).item()
        mean = data_stats['targets_mean']
        std = data_stats['targets_std']
    except FileNotFoundError:
        print(f"Error: Data statistics file '{data_stats_file}' not found.")
        return
    except KeyError as e:
        print(f"Error: Missing key '{e}' in data statistics file '{data_stats_file}'. Ensure targets_mean and targets_std are saved.")
        return
    except Exception as e:
        print(f"Error reading data statistics file: {e}")
        return

    predictions_df['true_values_standardized'] = predictions_df['True']
    predictions_df['predicted_values_standardized'] = predictions_df['Predicted']

    predictions_df['absolute_deviation'] = np.abs(predictions_df['true_values_standardized'] - predictions_df['predicted_values_standardized'])

    deviant_samples = []

    for sample_set_name, ratio in [('Train', train_ratio), ('Validation', other_ratio), ('Test', other_ratio)]:
        subset_df = predictions_df[predictions_df['Data Set'] == sample_set_name].copy()
        
        if not subset_df.empty:
            subset_df_sorted = subset_df.sort_values(by='absolute_deviation', ascending=False)
            
            num_to_select = int(len(subset_df_sorted) * ratio)
            
            if num_to_select == 0 and len(subset_df_sorted) > 0:
                num_to_select = 1
            
            selected_samples = subset_df_sorted.head(num_to_select)
            
            for index, row in selected_samples.iterrows():
                deviant_samples.append({
                    'battery_id': row.get('battery_id', ''),
                    'Formula': row['Formula'],
                    'True Value': row['true_values_standardized'], 
                    'Predicted Value': row['predicted_values_standardized'], 
                    'Data Set': row['Data Set'],
                    'Deviation': row['absolute_deviation'] 
                })

    deviant_df = pd.DataFrame(deviant_samples)

    if not deviant_df.empty:
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        deviant_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Deviant samples saved to '{output_file}'")
    else:
        print("No deviant samples found.")
