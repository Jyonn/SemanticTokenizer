import datetime
import json
import os
import sys
import time
from typing import cast

import numpy as np
import pigmento
import torch
from oba import Obj
from pigmento import pnt
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from loader.config_manager import ConfigManager
from loader.global_setting import Setting, Status
from model.it import ITOutput, IT, Handler
from utils.config_init import ConfigInit
from utils.gpu import GPU
from utils.meaner import Meaner
from utils.monitor import Monitor
from utils.structure import Structure


default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"1"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Worker:
    def __init__(self, config):
        self.config = config

        self.data, self.embed, self.model, self.exp = \
            self.config.data, self.config.embed, self.config.model, self.config.exp
        self.disable_tqdm = self.exp.policy.disable_tqdm
        self.mode = self.exp.mode

        self.init_pigmento()

        pnt('START TIME:', datetime.datetime.now())
        pnt(' '.join(sys.argv))
        pnt(json.dumps(Obj.raw(self.config), indent=4))

        Setting.device = self.get_device()

        self.config_manager = ConfigManager(
            data=self.data,
            embed=self.embed,
            model=self.model,
            exp=self.exp,
        )

        self.it = self.config_manager.it  # type: IT
        self.it.to(Setting.device)

        self.m_optimizer = None
        self.m_scheduler = None
        self.load_path = self.parse_load_path()

        pnt(self.config_manager.item_depot.depot[0])
        pnt(Structure().analyse_and_stringify(self.config_manager.sets.a_set()[0]))

    def init_pigmento(self):
        pigmento.add_time_prefix()
        pigmento.add_log_plugin(self.exp.log)
        pigmento.add_dynamic_color_plugin()
        pnt.set_display_mode(
            display_method_name=False,
            display_class_name=True,
        )

    def load(self, path):
        while True:
            pnt(f"load model from exp {path}")
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

    def get_device(self):
        cuda = self.config.cuda
        if cuda in ['-1', -1] or cuda is False:
            pnt('choose cpu')
            return 'cpu'
        if isinstance(cuda, int) or isinstance(cuda, str):
            pnt(f'User select cuda {cuda}')
            return f"cuda:{cuda}"
        return GPU.auto_choose(torch_format=True)

    def parse_load_path(self):
        if not self.exp.load.save_dir:
            return

        save_dir = os.path.join(self.exp.dir, self.exp.load.save_dir)
        save_dir = cast(str, save_dir)
        epochs = Obj.raw(self.exp.load.epochs)
        if not epochs:
            epochs = json.load(open(os.path.join(save_dir, 'candidates.json')))
        elif isinstance(epochs, str):
            epochs = eval(epochs)
        assert isinstance(epochs, list), ValueError(f'fail loading epochs: {epochs}')

        return [os.path.join(save_dir, f'epoch_{epoch}.bin') for epoch in epochs]

    def log_interval(self, epoch, step, loss_dict):
        line = ', '.join([f'{metric} {loss_dict[metric]:.4f}' for metric in loss_dict])
        pnt(f'[epoch {epoch}] step {step}, {line}')

    def log_epoch(self, epoch, loss_dict):
        line = ', '.join([f'{metric} {loss_dict[metric]:.4f}' for metric in loss_dict])
        pnt(f'[epoch {epoch}] {line}')

    def log_test(self, results):
        line = ', '.join([f'{metric} {results[metric]:.4f}' for metric in results])
        pnt(f'[test] {line}')

    def init(self):
        return
        if self.it.config.handler == Handler.BASELINE:
            return
        loader = self.config_manager.get_loader(Status.TRAIN)
        with torch.no_grad():
            embeds = []
            for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
                embed = self.it.init(batch=batch).detach()
                embeds.append(embed)
            embeds = torch.cat(embeds, dim=0)

            self.it.quantizer.initialize(embeds=embeds)
            self.it.to(Setting.device)

    def train(self) -> int:
        monitor_kwargs = Obj.raw(self.exp.store)

        monitor = Monitor(
            save_dir=self.exp.dir,
            **monitor_kwargs,
        )

        train_steps = len(self.config_manager.sets.train_set) // self.exp.policy.batch_size
        accumulate_step = 0
        accumulate_batch = self.exp.policy.accumulate_batch or 1

        self.init()

        loader = self.config_manager.get_loader(Status.TRAIN)
        self.m_optimizer.zero_grad()
        for epoch in range(self.exp.policy.epoch_start, self.exp.policy.epoch + self.exp.policy.epoch_start):
            # loader.start_epoch(epoch - self.exp.policy.epoch_start, self.exp.policy.epoch)
            self.it.quantizer.epoch_initialize()
            self.it.train()
            for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
                res = self.it(batch=batch)  # type: ITOutput
                loss = (res.generation_loss * self.exp.policy.gen_weight +
                        res.quantization_loss * self.exp.policy.quant_weight +
                        res.reconstruction_loss * self.exp.policy.recon_weight +
                        res.kl_divergence * self.exp.policy.kl_weight)
                # pnt(res.generation_loss, res.quantization_loss, res.reconstruction_loss, res.kl_divergence)
                loss.backward()

                accumulate_step += 1
                if accumulate_step == accumulate_batch:
                    self.m_optimizer.step()
                    self.m_scheduler.step()
                    self.m_optimizer.zero_grad()
                    accumulate_step = 0

                if self.exp.policy.check_interval:
                    loss_dict = res.voc_loss
                    loss_dict.update(dict(
                        gen_loss=res.generation_loss.item(),
                        quant_loss=res.quantization_loss.item(),
                        recon_loss=res.reconstruction_loss.item(),
                        kl=res.kl_divergence.item(),
                    ))
                    if self.exp.policy.check_interval < 0:  # step part
                        if (step + 1) % max(train_steps // (-self.exp.policy.check_interval), 1) == 0:
                            self.log_interval(epoch, step, loss_dict=loss_dict)
                    else:
                        if (step + 1) % self.exp.policy.check_interval == 0:
                            self.log_interval(epoch, step, loss_dict=loss_dict)
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

        pnt('Training Ended')
        monitor.export()

        return monitor.get_best_epoch()

    def evaluate(self, status):
        loader = self.config_manager.get_loader(status)

        total_loss = Meaner()
        quant_loss = Meaner()
        gen_loss = Meaner()
        recon_loss = Meaner()
        kl = Meaner()

        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                res = self.it(batch=batch)  # type: ITOutput
                # loss = res.quantization_loss * self.exp.policy.quant_weight + res.generation_loss
                loss = (res.generation_loss +
                        res.quantization_loss * self.exp.policy.quant_weight +
                        res.reconstruction_loss * self.exp.policy.recon_weight +
                        res.kl_divergence * self.exp.policy.kl_weight)
            total_loss.add(loss.item())
            quant_loss.add(res.quantization_loss.item())
            gen_loss.add(res.generation_loss.item())

            recon_loss.add(res.reconstruction_loss.item())
            kl.add(res.kl_divergence.item())

        total_loss = total_loss.mean()
        quant_loss = quant_loss.mean()
        gen_loss = gen_loss.mean()
        recon_loss = recon_loss.mean()
        kl = kl.mean()
        return dict(
            loss=total_loss,
            quant_loss=quant_loss,
            gen_loss=gen_loss,
            recon_loss=recon_loss,
            kl=kl,
        ), total_loss

    def dev(self):
        return self.evaluate(Status.DEV)

    def test(self):
        results, _ = self.evaluate(Status.TEST)
        self.log_test(results)

    def visualize(self):
        universal_decode = self.config_manager.embedding_manager.universal_decode
        loader = self.config_manager.get_loader(Status.TRAIN)
        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            res: ITOutput = self.it(batch=batch, visualize=True)
            true_labels, pred_labels = res.true_labels, res.pred_labels
            batch_size = true_labels.shape[0]
            # states = res.states
            # indices = res.indices
            for i_batch in range(min(batch_size, 10)):
                pnt(f'true: {universal_decode(true_labels[i_batch])}')
                pnt(f'pred: {universal_decode(pred_labels[i_batch])}')
                # pnt(f'indices: {indices[i_batch]}')
                # pnt(f'states: {states[i_batch]}')
                pnt('')

            break

    def train_runner(self):
        param_set = set()
        self.m_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.it.parameters()),
            lr=self.exp.policy.lr
        )
        self.m_scheduler = get_linear_schedule_with_warmup(
            self.m_optimizer,
            num_warmup_steps=self.exp.policy.n_warmup,
            num_training_steps=len(
                self.config_manager.sets.train_set) // self.exp.policy.batch_size * self.exp.policy.epoch,
        )

        for name, p in self.it.named_parameters():  # type: str, torch.Tensor
            if p.requires_grad:
                # param_key = '.'.join(name.split('.')[:2])
                # if param_key not in param_set:
                #     param_set.add(param_key)
                pnt(f'param {name} with shape {p.shape}')

        if self.load_path:
            self.load(self.load_path[0])
        return self.train()

    def iter_runner(self, handler):
        if self.load_path:
            for path in self.load_path:
                self.load(path)
                handler()
        else:
            handler()

    def display(self):
        pnt(self.config_manager.sets.a_set()[0])

    def export_states(self):
        store_dir = self.exp.store.export_dir

        num_items = len(self.config_manager.item_depot.depot)
        item_embeds = np.zeros((num_items, self.it.config.embed_dim), dtype=np.float32)

        with torch.no_grad():
            for status in [Status.TRAIN, Status.DEV, Status.TEST]:
                loader = self.config_manager.get_loader(status)
                for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
                    item_ids, embeds = self.it.get_embeds(batch=batch)
                    item_ids = item_ids.detach().cpu().numpy()
                    embeds = embeds.cpu().numpy()
                    item_embeds[item_ids] = embeds

        np.save(os.path.join(store_dir, 'item_embeds.npy'), item_embeds)

        pnt(f'export to {store_dir}')

    def export(self):
        store_dir = self.exp.store.export_dir

        codebooks = self.it.get_codebooks()
        # codebooks = codebooks.detach().cpu().numpy()
        np.save(os.path.join(store_dir, 'codebooks.npy'), codebooks)

        num_items = len(self.config_manager.item_depot.depot)
        if self.it.is_residual_vq:
            num_heads = self.it.config.residual_depth
        else:
            num_heads = self.it.config.num_heads
        code_matrix = np.zeros((num_items, num_heads), dtype=np.int32) - 1

        with torch.no_grad():
            for status in [Status.TRAIN, Status.DEV, Status.TEST]:
                loader = self.config_manager.get_loader(status)
                for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
                    item_ids, codes = self.it.get_codes(batch=batch)
                    item_ids = item_ids.detach().cpu().numpy()
                    codes = codes.cpu().numpy()
                    code_matrix[item_ids] = codes

        np.save(os.path.join(store_dir, 'codes.npy'), code_matrix)

        pnt(f'export to {store_dir}')

    def run(self):
        if self.mode == 'train':
            self.train_runner()
        elif self.mode == 'train_test':
            best_epoch = self.train_runner()
            pnt('best epoch', best_epoch)
            self.load(os.path.join(self.exp.dir, f'epoch_{best_epoch}.bin'))
            self.test()
        elif self.mode == 'test':
            self.iter_runner(self.test)
        elif self.mode == 'display':
            self.display()
        elif self.mode == 'export':
            self.load(self.load_path[0])
            self.export()
        elif self.mode == 'export_states':
            self.load(self.load_path[0])
            self.export_states()
        elif self.mode == 'visualize':
            self.load(self.load_path[0])
            self.visualize()


if __name__ == '__main__':
    configuration = ConfigInit(
        required_args=['data', 'exp'],
        default_args=dict(
            model='config/model/it.yaml',
            embed='config/embed/bart.yaml',
            warmup=0,
            batch_size=64,
            lr=0.001,
            patience=2,
            epoch_start=0,
            frozen=True,
            load_path=None,
            acc_batch=1,
            dim=768,
            pretrain=1,
            attnlayers=6,
            attnheads=12,

            quant=1.0,
            recon=1.0,
            kl=1.0,
            gen=1.0,
        ),
        makedirs=[
            'exp.dir',
        ]
    ).parse()

    worker = Worker(config=configuration)
    worker.run()
