import pandas as pd
import yaml
import os
#import cv2                  #conda install conda-forge::opencv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import PIL.Image as Image

    
train_path="./train"
val_path="./valid"

print("current path", os.getcwd())

print(train_path)

def get_files(path, endswith):
    files = []
    for root, dirs, file in os.walk(path):
        for f in file:
            if f.endswith(endswith):
                files.append(f)
    return files

train_images=get_files(train_path+"/images", ".jpg")
valid_images=get_files(val_path+"/images", ".jpg")
train_labels=get_files(train_path+"/labels", ".txt")
valid_labels=get_files(val_path+"/labels", ".txt")

print(len(train_images), len(train_labels), len(valid_images), len(valid_labels))

import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def read_and_process_file(file_path):
    try:
        df = pd.read_csv(file_path, sep=" ", header=None)
        df.columns = ['class', 'x', 'y', 'w', 'h']
        df['image'] = os.path.basename(file_path).replace('.txt', '.jpg')
        df['class'] = df['class'].astype(int)
        return df
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()

def get_df_from_labels_multicore(path, labels):
    files = [path+"/"+file for file in labels]
    dfs = []
    with ProcessPoolExecutor() as executor:
        # Schedule the file reading and processing to be executed
        future_to_file = {executor.submit(read_and_process_file, file): file for file in files}
        
        for future in as_completed(future_to_file):
            try:
                df = future.result()
                dfs.append(df)
            except Exception as exc:
                print('%r generated an exception: %s' % (future_to_file[future], exc))
    #print("dfs", dfs)
    # Combine all DataFrames
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


read_and_process_file(train_path+"/labels/"+train_labels[24])

# Beispiel-Nutzung:
# Stelle sicher, dass `train_path` und `train_labels` definiert sind.

if __name__ == '__main__':
    df = get_df_from_labels_multicore(train_path + "/labels", train_labels)
    df
