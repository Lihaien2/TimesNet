from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_MultiStock

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'multi_stock': Dataset_MultiStock,  # 新增
}

import torch

def custom_collate_fn(batch):
    """
    支持 4/5/6 元组：
      - 4: (x, y, xm, ym)
      - 5: (x, y, xm, ym, sid)
      - 6: (x, y, xm, ym, sid, last_close)
    """
    elem_len = len(batch[0])

    def _stack_x_y_marks(batch_):
        seq_x, seq_y, seq_x_mark, seq_y_mark = zip(*[b[:4] for b in batch_])
        seq_x = torch.stack([torch.tensor(x) for x in seq_x])
        seq_y = torch.stack([torch.tensor(y) for y in seq_y])
        seq_x_mark = torch.stack([torch.tensor(x) for x in seq_x_mark])
        seq_y_mark = torch.stack([torch.tensor(y) for y in seq_y_mark])
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    if elem_len == 4:
        return _stack_x_y_marks(batch)

    elif elem_len == 5:
        seq_x, seq_y, seq_x_mark, seq_y_mark = _stack_x_y_marks(batch)
        seq_sid = torch.stack([torch.tensor(b[4], dtype=torch.long)
                                for b in batch])
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_sid

    elif elem_len == 6:
        seq_x, seq_y, seq_x_mark, seq_y_mark = _stack_x_y_marks(batch)
        seq_sid = torch.stack([torch.tensor(b[4], dtype=torch.long)
                                for b in batch])
        seq_lc = torch.stack([torch.tensor(b[5], dtype=torch.float32)
                               for b in batch])
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_sid, seq_lc

    else:
        raise ValueError(f"Unsupported batch tuple length: {elem_len}")

def data_provider(args, flag):
    Data = data_dict[args.data]
    print(f"[DataProvider] args.data = {args.data}, using class: {Data.__name__}")
    timeenc = 0 if args.embed != 'timeF' else 1

    # Only training data may be shuffled. Validation/test ordering must stay deterministic
    # so predictions can be aligned with dates and stock identifiers during evaluation.
    shuffle_flag = str(flag).lower() == 'train'
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )
    elif args.data == 'multi_stock':
        # ---- 新增：多股票面板数据分支 ----
        assert args.label_len == 0, \
            "multi_stock 任务必须设置 --label_len 0"

        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            stock_code_col='stock_code',  # 显式传入
            window_stride=10,  # 滑动窗口步长
            train_ratio=0.7,
            vali_ratio=0.1,
        )
        print(f'[{flag}] 样本数: {len(data_set)}, '
              f'特征数: {data_set.n_features}, '
              f'seq_len={args.seq_len}, pred_len={args.pred_len}')

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=0,
            drop_last=drop_last,
            collate_fn=custom_collate_fn,
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=0,
            drop_last=drop_last,
            collate_fn=custom_collate_fn
        )
        return data_set, data_loader
