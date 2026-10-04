import json
import numpy as np
from pydicom import dcmread

def main():
    with open("../local_data/train.json","r") as f:
        train = json.load(f)

    with open("../local_data/val.json","r") as f:
        val = json.load(f)

    npy_val = []
    npy_train = []
    for item in val:
        file_name = item['input']
        array = dcmread(file_name)
        array = array.pixel_array
        out_path = file_name.replace(".dcm",".npy")
        np.save(out_path, array.astype(np.uint8))
        tmp = {}
        tmp['input'] = out_path
        tmp['label'] = item['targets']
        npy_val.append(tmp)
    print("Val Done")

    for item in train:
        file_name = item['input']
        array = dcmread(file_name)
        array = array.pixel_array
        out_path = file_name.replace(".dcm",".npy")
        np.save(out_path, array.astype(np.uint8))
        tmp = {}
        tmp['input'] = out_path
        tmp['label'] = item['targets']
        npy_train.append(tmp)

    print("Train Done")
    with open("valnumpy.json","w") as f:
        json.dump(npy_val, f)
    with open("trainnumpy.json","w") as f:
        json.dump(npy_train, f)
    return 0

if __name__=="__main__":
    main()
