import datetime
import json
import os
import sys
import time

import torch
from oba import Obj
from tqdm import tqdm

from loader.config_manager import ConfigManager
from loader.global_setting import Setting, Status
from model.it import ITOutput
from utils.config_init import ConfigInit
from utils.gpu import GPU
from utils.logger import Logger
from utils.meaner import Meaner
from utils.monitor import Monitor
from utils.printer import printer, Color, Printer
from utils.structure import Structure


class Worker:
    def __init__(self, config):
        self.config = config

        self.data, self.embed, self.model, self.exp = \
            self.config.data, self.config.embed, self.config.model, self.config.exp
        self.disable_tqdm = self.exp.policy.disable_tqdm

        self.print = printer[('MAIN', '·', Color.CYAN)]
        Printer.logger = Logger(self.exp.log)
        self.print('START TIME:', datetime.datetime.now())
        self.print(' '.join(sys.argv))
        self.print(json.dumps(Obj.raw(self.config), indent=4))

        Setting.device = self.get_device()

        self.config_manager = ConfigManager(
            data=self.data,
            embed=self.embed,
            model=self.model,
            exp=self.exp,
        )

        self.it = self.config_manager.it

        self.m_optimizer = None
        self.m_scheduler = None
        self.load_path = self.parse_load_path()

    def load(self, path):
        while True:
            self.print(f"load model from exp {path}")
            try:
                state_dict = torch.load(path, map_location=Setting.device)
                break
            except Exception as e:
                if not self.exp.load.wait_load:
                    raise e
                time.sleep(60)

        # compatible to old version where each operator are wrapped with an encoder
        model_ckpt = dict()
        for key, value in state_dict['model'].items():
            model_ckpt[key.replace('operator.', '')] = value

        self.it.load_state_dict(model_ckpt, strict=self.exp.load.strict)
        if not self.exp.load.model_only:
            self.m_optimizer.load_state_dict(state_dict['optimizer'])
            self.m_scheduler.load_state_dict(state_dict['scheduler'])

        self.print(self.config_manager.item_depot[0])
        self.print(Structure().analyse_and_stringify(self.config_manager.sets.a_set()[0]))

    def get_device(self):
        cuda = self.config.cuda
        if cuda in ['-1', -1] or cuda is False:
            self.print('choose cpu')
            return 'cpu'
        if isinstance(cuda, int) or isinstance(cuda, str):
            self.print(f'User select cuda {cuda}')
            return f"cuda:{cuda}"
        return GPU.auto_choose(torch_format=True)

    def parse_load_path(self):
        if not self.exp.load.save_dir:
            return

        save_dir = os.path.join(self.exp.dir, self.exp.load.save_dir)
        epochs = Obj.raw(self.exp.load.epochs)
        if not epochs:
            epochs = json.load(open(os.path.join(save_dir, 'candidates.json')))
        elif isinstance(epochs, str):
            epochs = eval(epochs)
        assert isinstance(epochs, list), ValueError(f'fail loading epochs: {epochs}')

        return [os.path.join(save_dir, f'epoch_{epoch}.bin') for epoch in epochs]

    def log_interval(self, epoch, step, loss):
        self.print[f'epoch {epoch}'](f'step {step}, loss {loss:.4f}')

    def log_epoch(self, epoch, results):
        line = ', '.join([f'{metric} {results[metric]:.4f}' for metric in results])
        self.print[f'epoch {epoch}'](line)

    def train(self) -> int:
        monitor_kwargs = Obj.raw(self.exp.store)

        monitor = Monitor(
            save_dir=self.exp.dir,
            **monitor_kwargs,
        )

        train_steps = len(self.config_manager.sets.train_set) // self.exp.policy.batch_size
        accumulate_step = 0
        accumulate_batch = self.exp.policy.accumulate_batch or 1

        loader = self.config_manager.get_loader(Status.TRAIN)
        self.m_optimizer.zero_grad()
        for epoch in range(self.exp.policy.epoch_start, self.exp.policy.epoch + self.exp.policy.epoch_start):
            # loader.start_epoch(epoch - self.exp.policy.epoch_start, self.exp.policy.epoch)
            self.it.train()
            loader.train()
            for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
                res = self.it(batch=batch)  # type: ITOutput
                loss = res.quantization_loss * self.exp.policy.quant_weight + res.generation_loss
                loss.backward()

                accumulate_step += 1
                if accumulate_step == accumulate_batch:
                    self.m_optimizer.step()
                    self.m_scheduler.step()
                    self.m_optimizer.zero_grad()
                    accumulate_step = 0

                if self.exp.policy.check_interval:
                    if self.exp.policy.check_interval < 0:  # step part
                        if (step + 1) % max(train_steps // (-self.exp.policy.check_interval), 1) == 0:
                            self.log_interval(epoch, step, loss.item())
                    else:
                        if (step + 1) % self.exp.policy.check_interval == 0:
                            self.log_interval(epoch, step, loss.item())

                if self.exp.policy.epoch_batch:
                    if self.exp.policy.epoch_batch < 0:  # step part
                        if step > max(train_steps // (-self.exp.policy.epoch_batch), 1):
                            break
                    else:
                        if step > self.exp.policy.epoch_batch:
                            break

            dev_results, monitor_metric = self.dev()
            self.log_epoch(epoch, dev_results)

            state_dict = dict(
                model=self.it.state_dict(),
                optimizer=self.m_optimizer.state_dict(),
                scheduler=self.m_scheduler.state_dict(),
            )
            early_stop = monitor.push(
                epoch=epoch,
                metric=monitor_metric,
                state_dict=state_dict,
            )
            if early_stop == -1:
                return monitor.get_best_epoch()

        self.print('Training Ended')
        monitor.export()

        return monitor.get_best_epoch()

    def evaluate(self, status):
        loader = self.config_manager.get_loader(status)

        total_loss = Meaner()

        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                res = self.it(batch=batch)  # type: ITOutput
                loss = res.quantization_loss * self.exp.policy.quant_weight + res.generation_loss
            total_loss.add(loss.item())

        total_loss = total_loss.mean()
        return dict(loss=total_loss), total_loss

    def dev(self):
        return self.evaluate(Status.DEV)

    def test(self):
        return self.evaluate(Status.TEST)

    def run(self):
        pass


if __name__ == '__main__':
    configuration = ConfigInit(
        required_args=['data', 'model', 'exp', 'embed'],
        default_args=dict(
            warmup=0,
            batch_size=64,
            lr=0.0001,
            patience=2,
            epoch_start=0,
            frozen=True,
            load_path=None,
        ),
        makedirs=[
            'exp.dir',
        ]
    ).parse()

    worker = Worker(config=configuration)
    worker.run()

