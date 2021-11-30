from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from colorama import Back, Fore, Style, init
from torch.utils.data.dataset import Dataset

from gradattack.trainingpipeline import TrainingPipeline

from .dpsgd import DPSGDDefense
from .gradprune import GradPruneDefense
from .instahide import InstahideDefense
from .mixup import MixupDefense

init(autoreset=True)
INDENT = "\t"


class DefensePack:
    def __init__(self, args, logger=None):
        self.defense_params = {}
        self.parse_defense_params(args)
        self.logger = logger  # this might be useful for logging DP prarameters in the future

    def apply_defense(self, pipeline: TrainingPipeline):
        dataset = pipeline.datamodule

        if "InstaHideDefense" in self.defense_params.keys():
            if "MixupDefense" in self.defense_params.keys():
                print("Overriding Mixup with InstaHide")
            params = self.defense_params["InstaHideDefense"]
            self.instahide_defense = InstahideDefense(
                mix_dataset=dataset.train_set,
                klam=params["k"],
                upper_bound=params["c_2"],
                lower_bound=params["c_1"],
                device=torch.device(params["device"]),
                use_csprng=params["use_csprng"])
            self.instahide_defense.apply(pipeline)
        if ("MixupDefense" in self.defense_params.keys()
                and "InstaHideDefense" not in self.defense_params.keys()):
            params = self.defense_params["MixupDefense"]
            self.mixup_defense = MixupDefense(mix_dataset=dataset.train_set,
                                              klam=params["k"],
                                              upper_bound=params["c_2"],
                                              lower_bound=params["c_1"],
                                              device=torch.device(
                                                  params["device"]),
                                              use_csprng=params["use_csprng"])
            self.mixup_defense.apply(pipeline)
        if "GradPruneDefense" in self.defense_params.keys():
            params = self.defense_params["GradPruneDefense"]
            self.gradprune_defense = GradPruneDefense(prune_ratio=params["p"])
            self.gradprune_defense.apply(pipeline)
        if "DPSGDDefense" in self.defense_params.keys():
            params = self.defense_params["DPSGDDefense"]
            self.DPSGD_defense = DPSGDDefense(
                delta_list=params["delta_list"],
                mini_batch_size=params["mini_batch_size"],
                max_grad_norm=params["max_grad_norm"],
                noise_multiplier=params["noise_multiplier"],
                n_accumulation_steps=params["n_accumulation_steps"],
                sample_size=len(dataset.train_set),
                secure_rng=params["secure_rng"],
                max_epsilon=params["max_epsilon"],
            )
            self.DPSGD_defense.apply(pipeline)

    def parse_defense_params(self, args, verbose=True):
        if args.defense_DPSGD:
            self.defense_params["DPSGDDefense"] = {
                "delta_list": args.delta_list,
                "mini_batch_size": args.batch_size,
                "max_grad_norm": args.max_grad_norm,
                "noise_multiplier": args.noise_multiplier,
                "secure_rng": args.secure_rng,
                "n_accumulation_steps": args.n_accumulation_steps,
                "freeze_extractor": args.freeze_extractor,
                "max_epsilon": args.max_epsilon,
            }
        if args.defense_mixup:
            self.defense_params["MixupDefense"] = {
                "k": args.klam,
                "c_1": args.c_1,
                "c_2": args.c_2,
                "device": f"cuda:{args.gpuid}",
                "use_csprng": args.use_csprng
            }
        if args.defense_instahide:
            self.defense_params["InstaHideDefense"] = {
                "k": args.klam,
                "c_1": args.c_1,
                "c_2": args.c_2,
                "device": f"cuda:{args.gpuid}",
                "use_csprng": args.use_csprng
            }
        if args.defense_gradprune:
            self.defense_params["GradPruneDefense"] = {"p": args.p}

        if verbose:
            print(Style.BRIGHT + Fore.CYAN + "Applied defenses:")
            for defense in self.defense_params.keys():
                print(Style.BRIGHT + Fore.GREEN + INDENT + "Defense name:",
                      end=" ")
                print(Fore.GREEN + defense, end="\t")
                print(Style.BRIGHT + Fore.MAGENTA + "Parameters:", end=" ")
                if len(self.defense_params[defense]) > 0:
                    for key, val in self.defense_params[defense].items():
                        print(Fore.MAGENTA + f"{key}: {val}", end="\t")
                else:
                    print(Fore.MAGENTA + "None", end="\t")
                print()

    def get_defensepack_str(self):
        def get_param_str(paramname):
            return "".join([
                f"{str(k)}{{{str(v)}}}-"
                for k, v in self.defense_params[paramname].items()
            ])

        return "".join([
            f"{str(p)}-{get_param_str(p)}" for p in self.defense_params.keys()
        ])[:-1]

    def __str__(self) -> str:
        return self.get_defensepack_str()
