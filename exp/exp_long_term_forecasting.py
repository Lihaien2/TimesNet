from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        self.train_loss_history = []
        self.vali_loss_history = []
        self.financial_metrics = {}
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                if len(batch) == 6:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, _, _ = batch
                elif len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            self.train_loss_history.append(train_loss)
            self.vali_loss_history.append(vali_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        os.makedirs('./results', exist_ok=True)
        loss_path = f'./results/loss_history_{self.args.model_id}_{self.args.model}.json'
        with open(loss_path, 'w') as f:
            json.dump({
                'train': self.train_loss_history,
                'vali': self.vali_loss_history,
                'model': self.args.model,
                'model_id': self.args.model_id,
            }, f, indent=2)
        print(f'Loss 历史已保存: {loss_path}')

        return self.model

    def _evaluate(self, test_data, test_loader, flag_name='test'):
        """参数化评估：与主 test 逻辑完全一致，可对 val 或 test 运行。"""
        preds = []
        trues = []
        stock_ids = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_sid = batch
                    stock_ids.append(batch_sid.numpy())
                elif len(batch) == 6:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_sid, _ = batch
                    stock_ids.append(batch_sid.numpy())
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y.detach().cpu().numpy()

                # ★ 关键：反标准化，与主 test 一致
                if self.args.inverse:
                    shape = batch_y_np.shape
                    outputs_np = test_data.inverse_transform(
                        outputs_np.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                    batch_y_np = test_data.inverse_transform(
                        batch_y_np.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)

                preds.append(outputs_np)
                trues.append(batch_y_np)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        stock_ids_arr = np.concatenate(stock_ids, axis=0) if stock_ids else None

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        preds_1d = preds[:, -1, -1].flatten()
        trues_1d = trues[:, -1, -1].flatten()

        from utils.metrics import (compute_direction_accuracy_grouped,
                                   backtest_strategy_grouped)

        dir_acc = compute_direction_accuracy_grouped(
            preds_1d, trues_1d, stock_ids_arr, neutral_threshold=0.002
        )
        backtest = backtest_strategy_grouped(
            preds_1d, trues_1d, stock_ids_arr,
            signal_threshold=0.005,
            cost_per_trade=0.0025,
            risk_free_rate=0.0,
            trading_days=252,
            pred_horizon=10,
            decision_interval=10,
        )

        print(f'\n===== [{flag_name}] =====')
        print(f'  n_samples = {len(preds_1d)}')
        print(f'  MSE       = {mse:.6f}')
        print(f'  MAE       = {mae:.6f}')
        print(f'  Dir_Acc   = {dir_acc:.4f}')
        print(f'  Sharpe    = {backtest["sharpe_ratio"]:+.3f}')
        print(f'  cum_ret   = {backtest["cumulative_return"]:+.4f}')
        print(f'  n_trades  = {backtest["n_trades"]}')
        print(f'  trade_ratio = {backtest["trade_ratio"]:.4f}')

        return backtest


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        preds = []
        trues = []
        stock_ids = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_sid = batch
                    stock_ids.append(batch_sid.numpy())
                elif len(batch) == 6:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_sid, _ = batch
                    stock_ids.append(batch_sid.numpy())
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # target 是收益率，不标准化，inverse_transform 是恒等函数
                if self.args.inverse:
                    shape = batch_y.shape
                    outputs = test_data.inverse_transform(
                        outputs.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                    batch_y = test_data.inverse_transform(
                        batch_y.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)

                preds.append(outputs)
                trues.append(batch_y)

                if i % 20 == 0:
                    input_ = batch_x.detach().cpu().numpy()
                    gt = batch_y[0, :, -1]
                    pd_ = outputs[0, :, -1]
                    # 收益率没有绝对值意义，画图直接用收益率
                    try:
                        visual(gt, pd_, os.path.join(folder_path, str(i) + '.pdf'))
                    except Exception:
                        pass

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        stock_ids_arr = np.concatenate(stock_ids, axis=0) if stock_ids else None

        print('test shape:', preds.shape, trues.shape)
        if stock_ids_arr is not None:
            print(f'stock_ids 形状: {stock_ids_arr.shape}, '
                  f'唯一值数: {len(np.unique(stock_ids_arr))}')

        # ----（此时 preds/trues 都是收益率）----
        # ---- 通用指标（收益率空间只报 MSE/MAE/RMSE）----
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'收益率空间 MSE:{mse:.8f}, MAE:{mae:.8f}, RMSE:{rmse:.8f}')
        # MAPE 和 MSPE 在收益率空间不适用，忽略

        # ---- 金融指标 ----
        from utils.metrics import (compute_direction_accuracy_grouped,
                                   backtest_strategy_grouped)

        preds_1d = preds[:, -1, -1].flatten()
        trues_1d = trues[:, -1, -1].flatten()

        # 路径2：阈值参数
        SIGNAL_THRESHOLD = 0.005  # 0.5% 的预测收益率才交易
        COST_PER_TRADE = 0.0025  # 双边 0.25%

        if stock_ids_arr is not None:
            dir_acc = compute_direction_accuracy_grouped(
                preds_1d, trues_1d, stock_ids_arr,
                neutral_threshold=0.002
            )
            # ---- 从数据集中读取 stride ----
            PRED_HORIZON = 10  # ★ 与 fetch_sse50.py 的 shift(-N) 一致
            DECISION_INTERVAL = getattr(test_data, 'window_stride', 10)  # 从 Dataset 读取

            # 打印配置，便于核对
            print(f'\n[Backtest Config]')
            print(f'  pred_horizon       = {PRED_HORIZON}')
            print(f'  decision_interval  = {DECISION_INTERVAL}')
            print(f'  periods_per_year   = {252 / DECISION_INTERVAL:.2f}')
            print(f'  holding_days       = {PRED_HORIZON}')
            print(f'  cash_days          = {DECISION_INTERVAL - PRED_HORIZON}')

            backtest = backtest_strategy_grouped(
                preds_1d, trues_1d, stock_ids_arr,
                signal_threshold=SIGNAL_THRESHOLD,
                cost_per_trade=COST_PER_TRADE,
                risk_free_rate=0.0,
                trading_days=252,
                pred_horizon=PRED_HORIZON,
                decision_interval=DECISION_INTERVAL,
            )
        else:
            raise RuntimeError('缺少 stock_ids')

        # # ---- 阈值扫描 ----
        # print('\n===== 阈值扫描 =====')
        # print(f'{"threshold":>10s} {"sharpe":>10s} {"cum_ret":>10s} '
        #       f'{"trade_ratio":>12s} {"win_rate":>10s} {"n_trades":>10s}')
        # for th in [0.001, 0.002, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050]:
        #     bt = backtest_strategy_grouped(
        #         preds_1d, trues_1d, stock_ids_arr,
        #         signal_threshold=th,
        #         cost_per_trade=COST_PER_TRADE,
        #         risk_free_rate=0.0,
        #         trading_days=252
        #     )
        #     print(f'{th:>10.4f} {bt["sharpe_ratio"]:>10.3f} '
        #           f'{bt["cumulative_return"]:>10.3f} '
        #           f'{bt["trade_ratio"]:>12.4f} '
        #           f'{bt["win_rate"]:>10.3f} {bt["n_trades"]:>10d}')

        self.financial_metrics = {
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'Direction_Acc': float(dir_acc),
            'signal_threshold': SIGNAL_THRESHOLD,
            'cost_per_trade': COST_PER_TRADE,
            **backtest,
        }

        print('\n===== 金融指标（收益率预测 + 阈值信号）=====')
        for k, v in self.financial_metrics.items():
            if isinstance(v, float):
                print(f'  {k:22s}: {v:.6f}')
            else:
                print(f'  {k:22s}: {v}')

        # ---- 保存 ----
        os.makedirs('./results', exist_ok=True)

        metrics_path = f'./results/metrics_{self.args.model_id}_{self.args.model}.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.financial_metrics, f, indent=2)
        print(f'指标已保存: {metrics_path}')

        np.save(f'./results/preds_{self.args.model_id}_{self.args.model}.npy', preds_1d)
        np.save(f'./results/trues_{self.args.model_id}_{self.args.model}.npy', trues_1d)
        np.save(f'./results/stock_ids_{self.args.model_id}_{self.args.model}.npy',
                stock_ids_arr)

        if hasattr(test_data, 'code_to_id'):
            with open(f'./results/code_to_id_{self.args.model_id}_{self.args.model}.json',
                      'w') as f:
                json.dump(test_data.code_to_id, f)

        folder_path2 = './results/' + setting + '/'
        os.makedirs(folder_path2, exist_ok=True)
        np.save(folder_path2 + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path2 + 'pred.npy', preds)
        np.save(folder_path2 + 'true.npy', trues)

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{:.6f}, mae:{:.6f}, dir_acc:{:.4f}, sharpe:{:.4f}'.format(
            mse, mae, dir_acc, backtest['sharpe_ratio']))
        f.write('\n\n')
        f.close()

        # ---- val 评估（用同一套逻辑）----
        print('\n' + '=' * 60)
        print('验证集评估（与测试集使用同一套逻辑）')
        print('=' * 60)
        vali_data, vali_loader = self._get_data(flag='val')
        self._evaluate(vali_data, vali_loader, flag_name='val')

        # ---- 市场基准：全多头策略 ----
        print('\n' + '=' * 60)
        print('市场基准（全多头）')
        print('=' * 60)

        for flag_name in ['val', 'test']:
            ds, dl = self._get_data(flag=flag_name)
            trues_bench = []
            self.model.eval()
            with torch.no_grad():
                for batch in dl:
                    if len(batch) == 5:
                        _, by, _, _, _ = batch
                    else:
                        _, by, _, _ = batch
                    by_np = by.detach().cpu().numpy()
                    if self.args.inverse:
                        shape = by_np.shape
                        by_np = ds.inverse_transform(
                            by_np.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                    trues_bench.append(by_np)

            t1d_bench = np.concatenate(trues_bench)[:, -1, -1].flatten()
            cum_ret = np.prod(1 + t1d_bench) - 1
            mean_r = t1d_bench.mean()
            std_r = t1d_bench.std()

            periods_per_year = 252 / 10
            ann_ret = (1 + mean_r) ** periods_per_year - 1
            ann_vol = std_r * np.sqrt(periods_per_year)
            sharpe_bench = ann_ret / ann_vol if ann_vol > 0 else 0

            print(f'[{flag_name}]')
            print(f'  n_samples = {len(t1d_bench)}')
            print(f'  累计收益   = {cum_ret:+.4f}')
            print(f'  平均收益   = {mean_r:.5f}')
            print(f'  标准差     = {std_r:.5f}')
            print(f'  年化收益   = {ann_ret:+.4f}')
            print(f'  年化波动   = {ann_vol:.4f}')
            print(f'  Sharpe     = {sharpe_bench:+.3f}')

        return
