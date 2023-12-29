import pandas as pd
from pathlib import Path
from glob import glob
from tqdm.auto import tqdm
l = ['训练','测试']

for c in l:
    all_path = glob(f'/home/liruixin/workspace/gait_classification/data/步态严重程度分类/*/角度/{c}')

    all_data = pd.DataFrame()  # 创建一个空的DataFrame
    for num_file in tqdm(range(16)):
        files = Path(all_path[num_file])
        for file in files.rglob('*.csv'):
            data = pd.read_csv(str(file))
            data['label'] = int(file.parts[-4][-1])
            if int(file.parts[-4][-1]) > 3:
                data['label'] = int(file.parts[-4][-1])-1
            all_data = pd.concat([all_data, data], ignore_index=True)
            pass
    all_data.to_csv(f'/home/liruixin/workspace/gait_classification/data/data_collect_{c}.csv', index=False)