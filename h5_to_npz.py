import h5py
import numpy as np



import numpy as np
import h5py
from pathlib import Path

def _to_numeric_array(x):
    # 尝试把任意对象转 ndarray
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        return None

def npz_to_h5(npz_path, h5_path, use_groups_for_ragged=False):
    npz_path = Path(npz_path)
    h5_path = Path(h5_path)

    with np.load(npz_path, allow_pickle=True, mmap_mode='r') as data, \
         h5py.File(h5_path, 'w') as hf:
        for key in data.files:
            arr = data[key]

            # 规则数组：直接写
            if arr.dtype != object:
                hf.create_dataset(key, data=arr, compression="gzip")
                continue

            # object 情况
            obj_list = list(arr)
            num_list = []
            ok_all_numeric = True
            for item in obj_list:
                a = _to_numeric_array(item)
                if a is None or a.dtype == object:
                    ok_all_numeric = False
                    num_list.append(item)  # 保留原样
                else:
                    num_list.append(a)

            if ok_all_numeric:
                shapes = [x.shape for x in num_list]
                # 形状一致 → 规则化 + 统一 dtype
                if len(set(shapes)) == 1:
                    stacked = np.stack(num_list, axis=0).astype(np.float32, copy=False)
                    hf.create_dataset(key, data=stacked, compression="gzip")
                    continue

                # 形状不一致 → 变长写入或逐元素写
                if not use_groups_for_ragged:
                    # 变长写入：把每个元素展平成一维并记录原形状
                    dt = h5py.vlen_dtype(np.dtype('float32'))
                    flat = [x.astype(np.float32, copy=False).ravel() for x in num_list]
                    dset = hf.create_dataset(key, (len(flat),), dtype=dt, compression="gzip")
                    dset[:] = flat
                    # 额外存形状，便于还原
                    shapes_dset = hf.create_dataset(f"{key}__shapes", data=np.array(
                        [np.array(s, dtype=np.int64) for s in shapes], dtype=object
                    ), dtype=h5py.vlen_dtype(np.dtype('int64')))
                    continue
                else:
                    grp = hf.create_group(key)
                    for i, x in enumerate(num_list):
                        grp.create_dataset(str(i), data=x.astype(np.float32, copy=False), compression="gzip")
                    continue

            # 混杂非数值对象：逐元素子 dataset 或转字符串
            grp = hf.create_group(key)
            for i, item in enumerate(obj_list):
                name = str(i)
                a = _to_numeric_array(item)
                if isinstance(a, np.ndarray) and a.dtype != object:
                    grp.create_dataset(name, data=a.astype(np.float32, copy=False), compression="gzip")
                elif isinstance(item, (bytes, bytearray)):
                    dt = h5py.vlen_dtype(np.dtype('uint8'))
                    grp.create_dataset(name, data=np.frombuffer(item, dtype='uint8'), dtype=dt)
                else:
                    dt = h5py.string_dtype(encoding='utf-8')
                    grp.create_dataset(name, data=str(item), dtype=dt)

if __name__ == "__main__":
    npz_to_h5(
        "./data/xian/eval.npz",
        "./data/xian/eval.h5",
        use_groups_for_ragged=False  # 想逐元素子dataset就改为 True
    )
