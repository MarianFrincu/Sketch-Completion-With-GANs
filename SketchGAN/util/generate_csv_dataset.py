import os
import pandas as pd

original_dir = '../datasets/Sketchy/original/train'
corrupted_dir = '../datasets/Sketchy/corrupted/train'
csv_output_path = '../datasets/Sketchy/train_labels.csv'

data = []

for class_name in os.listdir(original_dir):
    class_dir = os.path.join(original_dir, class_name)
    if os.path.isdir(class_dir):
        for image_name in os.listdir(class_dir):
            if image_name.endswith(('.png', '.jpg', '.jpeg')):
                data.append([image_name, class_name])

df = pd.DataFrame(data, columns=['image_name', 'class_name'])

df.to_csv(csv_output_path, index=False)
