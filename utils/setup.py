import torch
import random
import numpy as np
from loguru import logger

import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM
from .modeling_llama import LlamaForCausalLM
from .lions import Lion as Lion_we

# import bitsandbytes as bnb
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor, QGaLoreAdamW8bit, QGaLoreAdamW8bit_simulate, SPAM,StableSPAM,Adam_mini_our

from .training_utils import get_scheculer

from .fake_quantization import QLinear, prepare_model_for_int8_training_simulation
from .quantization import QScaleLinear, prepare_model_for_int8_training
from .act_weight_quantization import prepare_model_for_int8_training_simulation_act_weight
from .act_weight_fp4 import prepare_model_for_fp4_training_simulation_act_weight
from .real_fp4_quantization import prepare_model_for_real_fp4_training_simulation_act_weight

def getting_svd_cnt(optimizer):
    svd_cnt = 0
    # for key in optimizer.state_dict()['state']:
    #     if 'projector' in optimizer.state_dict()['state'][key]:
    #         if hasattr(optimizer.state_dict()['state'][key]['projector'], 'svd_count'):
    #             svd_cnt += optimizer.state_dict()['state'][key]['projector'].svd_count
    return svd_cnt

def set_seed(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def setup_model(args):
    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    # if args.weight_quant:
    #     # assert args.optimizer.lower() in ['q_galore_adamw8bit', 'q_galore_adamw8bit_per_layer']
    #     print('Quantizing weight')
    #     target_module = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
    #     if args.simulation:
    #         print('Using Simulation Mode')
    #         model = prepare_model_for_int8_training_simulation(model, args, target_module)
    #     else:
    #         model = prepare_model_for_int8_training(model, args, target_module)
    #     logger.info('--'*20)
    #     logger.info('Prepare Model for Int8 Training')
    #     logger.info('--'*20)
    if args.act_quant:
        print('Activation-Weight Quantizing')
        target_module = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
        module = prepare_model_for_int8_training_simulation_act_weight(model,args,target_module)
        logger.info('--'*20)
        logger.info('Prepare Model for Activation&Weight Int8 Training')
        logger.info('--'*20)
    if args.fp4 :
        print('FP4 training')
        target_module = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
        module = prepare_model_for_fp4_training_simulation_act_weight(model,args,target_module)
        logger.info('--'*20)
        logger.info('Prepare Model for Activation&Weight FP4 Training')
        logger.info('--'*20)
    if args.real_fp4:
        print('Real FP4 training')
        target_module = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
        module = prepare_model_for_real_fp4_training_simulation_act_weight(model,args,target_module)
        logger.info('--'*20)
        logger.info('Prepare Model for Real FP4 Training')
        logger.info('--'*20)   

    return model_config, model


def saving_model_weight(model, path, args):
    """
    Save model weight to file
    """
    checkpoint = model.state_dict()
    if args.simulation and args.weight_quant:
        for name, module in model.named_modules():
            if isinstance(module, QLinear):
                checkpoint[name + '.weight'] = module.weight
                if module.bias is not None:
                    checkpoint[name + '.bias'] = module.bias
                checkpoint[name + '.group_size'] = module.weight.group_size
                checkpoint[name + '.stochastic_round'] = module.weight.stochastic_round
                checkpoint[name + '.num_bits'] = module.num_bits
                checkpoint[name + '.group_size'] = module.group_size

    elif args.weight_quant:
        for name, module in model.named_modules():
            if isinstance(module, QScaleLinear):
                checkpoint[name + '.weight'] = module.weight
                if module.bias is not None:
                    checkpoint[name + '.bias'] = module.bias
                checkpoint[name + '.scales'] = module.weight.scales
                checkpoint[name + '.zeros'] = module.weight.zeros
                checkpoint[name + '.group_size'] = module.weight.group_size
                checkpoint[name + '.saved_data_dtype'] = module.weight.saved_data_dtype
                checkpoint[name + '.stochastic_round'] = module.weight.stochastic_round
    else:
        print('saving model weight without quantized layer')
    torch.save(checkpoint, path)

def load_model_weight(model, path, args):
    """
    Load model weight from file
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    if args.simulation and args.weight_quant:
        for name, module in model.named_modules():
            if isinstance(module, QLinear):
                module.weight = checkpoint[name + '.weight']
                if module.bias is not None:
                    module.bias = checkpoint[name + '.bias']
                module.weight.group_size = checkpoint[name + '.group_size']
                module.weight.stochastic_round = checkpoint[name + '.stochastic_round']
                module.num_bits = checkpoint[name + '.num_bits']
                module.group_size = checkpoint[name + '.group_size']

    elif args.weight_quant:
        for name, module in model.named_modules():
            if isinstance(module, QScaleLinear):
                module.weight = checkpoint[name + '.weight']
                if module.bias is not None:
                    module.bias = checkpoint[name + '.bias']
                module.weight.scales = checkpoint[name + '.scales']
                module.weight.zeros = checkpoint[name + '.zeros']
                module.weight.group_size = checkpoint[name + '.group_size']
                module.weight.saved_data_dtype = checkpoint[name + '.saved_data_dtype']
                module.weight.stochastic_round = checkpoint[name + '.stochastic_round']
    else:
        print('loading model weight without quantized layer')
    return model

def setup_optimization(args, model, trainable_params, param_groups, id_galore_params,model_config=None):
    layer_wise_flag = False
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower()=='adammini':
        from adam_mini import Adam_mini
        optimizer = Adam_mini(
            named_parameters = model.named_parameters(), 
            lr = args.lr, 
            weight_decay = args.weight_decay, 
            model_sharding = True,
            dim = model_config.hidden_size,
            n_heads = model_config.num_attention_heads,
            )
    elif args.optimizer.lower()=='adam_mini_our':
        optimizer = Adam_mini_our(
            named_parameters = model.named_parameters(), 
            lr = args.lr, 
            weight_decay = args.weight_decay, 
            model_sharding = True,
            dim = model_config.hidden_size,
            n_heads = model_config.num_attention_heads,
            gamma1 = args.gamma1,
            gamma2 = args.gamma2,
            theta = args.gamma3,
            )
    elif args.optimizer.lower()=='lion':
        from lion_pytorch import Lion
        optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower()=='lion_our':
        optimizer = Lion_we(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,gamma1 = args.gamma1, gamma2 = args.gamma2, theta = args.gamma3)
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_adamw":
        optimizer = GaLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)    
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.beta1)
    elif args.optimizer.lower() == "spam":
        optimizer = SPAM(param_groups, lr = args.lr, weight_decay = args.weight_decay)
        ########################################################
    elif args.optimizer.lower() == "stablespam":
        optimizer = StableSPAM(trainable_params, lr = args.lr, weight_decay = args.weight_decay,gamma1=args.gamma1,gamma2=args.gamma2,gamma3=args.gamma3,eta_min=args.eta,update_proj_gap=args.update_proj_gap,total_T=args.total_T)
    elif args.optimizer.lower() == "adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = transformers.optimization.Adafactor(
            trainable_params,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    elif args.optimizer.lower() == "galore_adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

    # elif args.optimizer.lower() == "adam8bit":
    #     optimizer = bnb.optim.Adam8bit(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    elif args.optimizer.lower() == "galore_adamw8bit":
        optimizer = GaLoreAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))

    elif args.optimizer.lower() == "q_galore_adamw8bit":
        if args.simulation:
            print('Using Simulation Mode')
            optimizer = QGaLoreAdamW8bit_simulate(param_groups, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
        else:
            optimizer = QGaLoreAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))

    elif args.optimizer.lower() == 'galore_adamw8bit_per_layer':
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
        optimizer_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = GaLoreAdamW8bit([{'params': [p], 
                        'rank': args.rank, 'update_proj_gap': args.update_proj_gap * 2, 
                        'scale': args.galore_scale, 'proj_type': args.proj_type}], lr=args.lr, weight_decay=args.weight_decay)
                # else:
                #     optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=args.lr, weight_decay=args.weight_decay)

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if p.grad is None: 
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)

        layer_wise_flag = True

    elif args.optimizer.lower() == 'q_galore_adamw8bit_per_layer':
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
        optimizer_dict = {}
        for p in model.parameters():
            if id(p) in id_galore_params:
                optimizer_dict[p] = QGaLoreAdamW8bit([{'params': [p], 
                    'rank': args.rank, 'update_proj_gap': args.update_proj_gap * 2, 
                    'scale': args.galore_scale, 'proj_type': args.proj_type,
                    "quant": args.proj_quant,'quant_n_bit': args.proj_bits, 'quant_group_size': args.proj_group_size,
                    'cos_threshold': args.cos_threshold, 'gamma_proj': args.gamma_proj, 'queue_size': args.queue_size}], lr=args.lr, weight_decay=args.weight_decay)
            # else:
            #     if p.requires_grad:
            #         optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=args.lr, weight_decay=args.weight_decay)

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if id(p) in id_galore_params or p.requires_grad:
                scheduler_dict[p] = get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):

            if (not hasattr(p, 'float_grad')) and p.grad is None: 
                return

            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
        layer_wise_flag = True

    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    
    if not layer_wise_flag:
        scheduler = get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    return model, optimizer, scheduler, layer_wise_flag




