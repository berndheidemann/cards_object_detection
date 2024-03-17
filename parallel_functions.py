import os
import imageio
import pandas as pd
# use imageio

from PIL import Image



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
    
def read_image_and_bbox(chunck, train_path):
    image = imageio.imread(train_path+"/"+chunck.iloc[0]['image'])
    image = Image.fromarray(image)
    return {"image": image, "chunck": chunck}