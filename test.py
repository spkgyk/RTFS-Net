import os
import sys
import yaml
import torch
import argparse
import warnings
import importlib
import torchaudio
import pandas as pd
from tqdm import tqdm
from sigfig import round
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.metrics import ALLMetricsTracker
from src.utils.parser_utils import parse_args_as_dict
from src.datas.avspeech_dataset import AVSpeechDataset
from src.losses import PITLossWrapper, pairwise_neg_sisdr

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")


class TestModule(pl.LightningModule):
    def __init__(self, conf):
        super(TestModule, self).__init__()
        self.conf = conf
        self.conf["videonet"] = conf.get("videonet", {})
        self.conf["videonet"]["model_name"] = conf["videonet"].get("model_name", None)

        self.exp_dir = os.path.abspath(os.path.join("../experiments/audio-visual", conf["log"]["exp_name"]))

        sys.path.append(os.path.dirname(self.exp_dir))
        models_module = importlib.import_module(os.path.basename(self.exp_dir) + ".models")
        videomodels = importlib.import_module(os.path.basename(self.exp_dir) + ".models.videomodels")
        AVNet = getattr(models_module, "AVNet")

        model_path = os.path.join(self.exp_dir, "best_model.pth")
        self.audiomodel = AVNet.from_pretrain(model_path, **self.conf["audionet"])
        self.videomodel = None
        if self.conf["videonet"]["model_name"]:
            self.videomodel = videomodels.get(self.conf["videonet"]["model_name"])(**self.conf["videonet"], print_macs=False)

        self.loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

        self.ex_save_dir = os.path.join(self.exp_dir, "results_new/")
        os.makedirs(self.ex_save_dir, exist_ok=True)
        self.metrics = ALLMetricsTracker(save_file=os.path.join(self.ex_save_dir, "metrics.csv"))
        self.test_step_outputs = {"mix": [], "sources": [], "est_sources": [], "key": []}

    def test_step(self, batch, batch_idx):
        mix, sources, target_mouths, key = batch
        sources = sources.unsqueeze(1)
        mouth_emb = self.videomodel(target_mouths.float()) if self.videomodel is not None else None
        est_sources = self.audiomodel(mix, mouth_emb)
        loss, reordered_sources = self.loss_func(est_sources, sources, return_ests=True)
        self.log("test_loss", loss, prog_bar=True)
        self.test_step_outputs["mix"].append(mix)
        self.test_step_outputs["sources"].append(sources)
        self.test_step_outputs["est_sources"].append(reordered_sources)
        self.test_step_outputs["key"].append(key)
        return loss

    def on_test_epoch_end(self):
        with torch.no_grad():
            mix = [x for x in torch.cat(self.test_step_outputs["mix"], dim=0)]
            sources = [x for x in torch.cat(self.test_step_outputs["sources"], dim=0)]
            est_sources = [x for x in torch.cat(self.test_step_outputs["est_sources"], dim=0)]
            key = [item for sublist in self.test_step_outputs["key"] for item in sublist]
            pbar = tqdm(range(len(mix)))
            for idx in pbar:
                self.metrics(mix=mix[idx], clean=sources[idx], estimate=est_sources[idx], key=key[idx])
                if idx < self.conf["n_save_ex"]:
                    self._save_audio_example(idx, mix[idx], sources[idx], est_sources[idx])
                if not (idx % 10):
                    pbar.set_postfix(self.metrics.get_mean())

            self.metrics.final()
            mean, std = self.metrics.get_mean(), self.metrics.get_std()
            keys = list(mean.keys() & std.keys())

            order = ["si-snr_i", "sdr_i", "pesq", "stoi", "si-snr", "sdr"]

            def get_order(k):
                try:
                    ind = order.index(k)
                    return ind
                except ValueError:
                    return 100

            self.audiomodel.get_MACs()
            self.videomodel.get_MACs()

            results_dict = []
            results_dict.append(("Model", self.conf["log"]["exp_name"]))
            results_dict.append(("MACs and Params", self.audiomodel.macs_parms))
            results_dict.append(("Videomodel MACs", self.videomodel.macs))
            results_dict.append(("Videomodel Params", self.videomodel.number_of_parameters))

            keys.sort(key=get_order)
            for k in keys:
                m, s = round(mean[k], 4), round(std[k], 3)
                results_dict.append((k, str(m) + " ± " + str(s)))
                print(f"{k}\tmean: {m}  std: {s}")

            for k, v in self.conf["audionet"].items():
                if isinstance(v, dict):
                    results_dict.extend([(k + "_" + kk, vv) for kk, vv in v.items()])
                else:
                    results_dict.append((k, v))

            df = pd.DataFrame.from_records(results_dict, columns=["Key", "Value"])
            df.to_csv(os.path.join(self.ex_save_dir, "results.csv"), encoding="utf-8", index=False)

    def _save_audio_example(self, idx, mix_np, sources_np, est_sources_np):
        examples_dir = os.path.join(self.ex_save_dir, "examples")
        if not os.path.exists(examples_dir):
            os.makedirs(examples_dir)

        est_sources_np = est_sources_np[0].cpu().unsqueeze(0)
        torchaudio.save(os.path.join(self.ex_save_dir, "examples", str(idx) + "_est.wav"), est_sources_np, 16000)
        sources_np = sources_np[0].cpu().unsqueeze(0)
        torchaudio.save(os.path.join(self.ex_save_dir, "examples", str(idx) + "_gt.wav"), sources_np, 16000)
        mix_np = mix_np.cpu().unsqueeze(0)
        torchaudio.save(os.path.join(self.ex_save_dir, "examples", str(idx) + "_mix.wav"), mix_np, 16000)

    def test_dataloader(self):
        test_set = AVSpeechDataset(
            self.conf["test_dir"],
            n_src=self.conf["data"]["nondefault_nsrc"],
            sample_rate=self.conf["data"]["sample_rate"],
            segment=None,
            normalize_audio=self.conf["data"]["normalize_audio"],
        )
        data_loader = DataLoader(
            test_set,
            shuffle=False,
            batch_size=self.conf["training"]["batch_size"] * 2,
            num_workers=self.conf["training"]["num_workers"],
        )
        return data_loader


def main(conf):
    model = TestModule(conf)
    trainer = pl.Trainer(
        default_root_dir=model.exp_dir,
        devices=[0],
        accelerator="auto",
        sync_batchnorm=True,
    )
    trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--test-dir",
        type=str,
        default="data-preprocess/LRS2/tt",
        help="Test directory including the json files",
    )
    parser.add_argument(
        "-c",
        "--conf-dir",
        type=str,
        default="../experiments/audio-visual/RTFS-Net/LRS2/4_layers/conf.yaml",
        help="Full path to save best validation model",
    )
    parser.add_argument(
        "--n-save-ex",
        type=int,
        default=-1,
        help="Number of audio examples to save, -1 means none",
    )

    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic["main_args"])

    main(def_conf)
