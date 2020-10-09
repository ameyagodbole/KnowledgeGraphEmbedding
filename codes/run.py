#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel
from stream_utils import KBStream
from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('--stream_init_proportion', type=float, default=0.5)
    parser.add_argument('--n_stream_updates', type=int, default=10)
    parser.add_argument('--frac_old_train_samples', type=float, default=0.1)
    parser.add_argument('--sample_nbh', action='store_true')
    parser.add_argument('--stream_seed', type=int, default=42)
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--ft_all_entity_embeddings', action='store_true')
    parser.add_argument('--smart_init', action='store_true')
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--ft_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    # parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args, stream_step):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, f'config_{stream_step}.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, f'checkpoint_stream_step_{stream_step}')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, f'entity_embedding_{stream_step}'),
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, f'relation_embedding_{stream_step}'),
        relation_embedding
    )


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        

def main_ops(args, entity2id, relation2id, all_true_triples, new_train_triples, all_valid_triples, new_valid_triples,
             all_test_triples, new_test_triples, nentity_old, nentity_new, batch_step):
    logging.info(f'[MAIN OPS] Stream step {batch_step}')
    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    kge_model = KGEModel(
        model_name=args.model,
        nentity_old=nentity_old,
        nentity_new=nentity_new,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        train_old_e=args.ft_all_entity_embeddings
    )

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(new_train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(new_train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2 if batch_step == 0 else args.ft_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, f'checkpoint_stream_step_{batch_step}'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    elif batch_step > 0:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % os.path.join(args.save_path, f'checkpoint_stream_step_{batch_step-1}'))
        device = torch.device('cuda') if args.cuda else torch.device('cpu')
        checkpoint = torch.load(os.path.join(args.save_path, f'checkpoint_stream_step_{batch_step-1}'),
                                    map_location=device)
        init_step = checkpoint['step']
        # Adjust for changed vocab size
        # if nentity > checkpoint['model_state_dict']['entity_embedding'].shape[0]:

        k_oe, k_ne, k_r = None, None, None
        old_e_embed_shape, new_e_embed_shape = checkpoint['model_state_dict']['entity_embedding_old'].shape, \
                                               checkpoint['model_state_dict']['entity_embedding_new'].shape
        old_r_embed_shape = checkpoint['model_state_dict']['relation_embedding'].shape
        for _k, _v in checkpoint['optimizer_state_dict']['state'].items():
            if _v['exp_avg'].shape == old_r_embed_shape:
                k_r = _k
                continue
            if _v['exp_avg'].shape == old_e_embed_shape:
                k_oe = _k
                continue
            if _v['exp_avg'].shape == new_e_embed_shape:
                k_ne = _k
                continue
            logging.warning(f"Unassigned key for tensot of size: {_v.shape}")
        logging.info(f"k_oe:{k_oe} k_ne:{k_ne} k_r:{k_r}")

        if nentity_new > 0:
            ###################
            # WORKING_VERSION #
            ###################
            # old_embed_shape = checkpoint['model_state_dict']['entity_embedding'].shape
            # print(f'{kge_model.entity_embedding.shape} {kge_model.entity_embedding.is_leaf} {kge_model.entity_embedding.requires_grad}')
            # kge_model.entity_embedding.data[:old_embed_shape[0]] = checkpoint['model_state_dict']['entity_embedding']
            # del checkpoint['model_state_dict']['entity_embedding']
            # print(f'{kge_model.entity_embedding.shape} {kge_model.entity_embedding.is_leaf} {kge_model.entity_embedding.requires_grad}')
            # if len(checkpoint['optimizer_state_dict']['state'].keys()) > 0:
            #     k1, k2 = checkpoint['optimizer_state_dict']['state'].keys()
            #     if checkpoint['optimizer_state_dict']['state'][k1]['exp_avg'].shape == old_embed_shape:
            #         temp_tensor = torch.zeros_like(kge_model.entity_embedding)
            #         temp_tensor[:old_embed_shape[0]] = checkpoint['optimizer_state_dict']['state'][k1]['exp_avg']
            #         checkpoint['optimizer_state_dict']['state'][k1]['exp_avg'] = temp_tensor.detach()
            #         temp_tensor = torch.zeros_like(kge_model.entity_embedding)
            #         temp_tensor[:old_embed_shape[0]] = checkpoint['optimizer_state_dict']['state'][k1]['exp_avg_sq']
            #         checkpoint['optimizer_state_dict']['state'][k1]['exp_avg_sq'] = temp_tensor.detach()
            #     elif checkpoint['optimizer_state_dict']['state'][k2]['exp_avg'].shape == old_embed_shape:
            #         temp_tensor = torch.zeros_like(kge_model.entity_embedding)
            #         temp_tensor[:old_embed_shape[0]] = checkpoint['optimizer_state_dict']['state'][k2]['exp_avg']
            #         checkpoint['optimizer_state_dict']['state'][k2]['exp_avg'] = temp_tensor.detach()
            #         temp_tensor = torch.zeros_like(kge_model.entity_embedding)
            #         temp_tensor[:old_embed_shape[0]] = checkpoint['optimizer_state_dict']['state'][k2]['exp_avg_sq']
            #         checkpoint['optimizer_state_dict']['state'][k2]['exp_avg_sq'] = temp_tensor.detach()
            #     else:
            #         raise ValueError("Tensor shape mismatch")
            ###################
            # WORKING_VERSION #
            ###################
            print(f'{kge_model.entity_embedding.shape} {kge_model.entity_embedding.is_leaf} {kge_model.entity_embedding.requires_grad}')
            kge_model.entity_embedding_old.data = torch.cat([checkpoint['model_state_dict']['entity_embedding_old'], checkpoint['model_state_dict']['entity_embedding_new']], dim=0)
            del checkpoint['model_state_dict']['entity_embedding_old']
            del checkpoint['model_state_dict']['entity_embedding_new']
            print(
                f'{kge_model.entity_embedding.shape} {kge_model.entity_embedding.is_leaf} {kge_model.entity_embedding.requires_grad}')

            if k_oe is not None and k_ne is not None:
                if args.ft_all_entity_embeddings:
                    temp_tensor = torch.zeros_like(kge_model.entity_embedding_old)
                    temp_tensor[:old_e_embed_shape[0]] = checkpoint['optimizer_state_dict']['state'][k_oe]['exp_avg']
                    temp_tensor[old_e_embed_shape[0]:] = checkpoint['optimizer_state_dict']['state'][k_ne]['exp_avg']
                    checkpoint['optimizer_state_dict']['state'][k_oe]['exp_avg'] = temp_tensor.detach()
                    temp_tensor = torch.zeros_like(kge_model.entity_embedding_old)
                    temp_tensor[:old_e_embed_shape[0]] = checkpoint['optimizer_state_dict']['state'][k_oe]['exp_avg_sq']
                    temp_tensor[old_e_embed_shape[0]:] = checkpoint['optimizer_state_dict']['state'][k_ne]['exp_avg_sq']
                    checkpoint['optimizer_state_dict']['state'][k_oe]['exp_avg_sq'] = temp_tensor.detach()
            if k_ne is not None:
                checkpoint['optimizer_state_dict']['state'][k_ne]['exp_avg'] = torch.zeros_like(kge_model.entity_embedding_new).detach()
                checkpoint['optimizer_state_dict']['state'][k_ne]['exp_avg_sq'] = torch.zeros_like(kge_model.entity_embedding_new).detach()

        nrelation_old = old_r_embed_shape[0]
        if nrelation > checkpoint['model_state_dict']['relation_embedding'].shape[0]:
            # new_r_embed = kge_model.relation_embedding
            # ckpt_r_embed = checkpoint['model_state_dict']['relation_embedding']
            # new_r_embed[:ckpt_r_embed.shape[0]] = ckpt_r_embed
            # checkpoint['model_state_dict']['relation_embedding'] = new_r_embed.detach().requires_grad_(True)
            print(f'{kge_model.relation_embedding.shape} {kge_model.relation_embedding.is_leaf} {kge_model.relation_embedding.requires_grad}')
            kge_model.relation_embedding.data[:nrelation_old] = checkpoint['model_state_dict']['relation_embedding']
            del checkpoint['model_state_dict']['relation_embedding']
            print(f'{kge_model.relation_embedding.shape} {kge_model.relation_embedding.is_leaf} {kge_model.relation_embedding.requires_grad}')

            if k_r is not None:
                temp_tensor = torch.zeros_like(kge_model.relation_embedding)
                temp_tensor[:nrelation_old] = checkpoint['optimizer_state_dict']['state'][k_r]['exp_avg']
                checkpoint['optimizer_state_dict']['state'][k_r]['exp_avg'] = temp_tensor.detach()
                temp_tensor = torch.zeros_like(kge_model.relation_embedding)
                temp_tensor[:nrelation_old] = checkpoint['optimizer_state_dict']['state'][k_r]['exp_avg_sq']
                checkpoint['optimizer_state_dict']['state'][k_r]['exp_avg_sq'] = temp_tensor.detach()

        kge_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if args.smart_init and nentity_new > 0:
            logging.info('Beginning smart init...')
            kge_model.entity_embedding_new.data = kge_model.edge_init(kge_model, new_train_triples, nrelation_old,
                                                                      nentity_old, nentity_new, args)
        if args.do_train:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate
            )
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    # Set valid dataloader as it would be evaluated during training

    max_steps = args.max_steps if batch_step == 0 else step + args.ft_steps
    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        best_mrr = 0.0
        # validation_cooldown = 5
        # Training Loop
        for step in range(init_step, max_steps):

            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)

            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            # if step % args.save_checkpoint_steps == 0:
            #     save_variable_list = {
            #         'step': step,
            #         'current_learning_rate': current_learning_rate,
            #         'warm_up_steps': warm_up_steps
            #     }
            #     save_model(kge_model, optimizer, save_variable_list, args, batch_step)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            if args.do_valid and (step % args.valid_steps == 0 or step == max_steps - 1):
                logging.info('Evaluating on Currently Known Valid Triples...')
                metrics = kge_model.test_step(kge_model, all_valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)
                if metrics['MRR'] >= best_mrr:
                    logging.info('Checkpointing the model...')
                    best_mrr = metrics['MRR']
                    # validation_cooldown = 5
                    save_variable_list = {
                        'step': step,
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    save_model(kge_model, optimizer, save_variable_list, args, batch_step)
                # else:
                #     validation_cooldown -= 1
                #     logging.info(f'Validation cooldown = {validation_cooldown}')
                #     if validation_cooldown == 0:
                #         break

        # Restore best checkpoint
        checkpoint = torch.load(os.path.join(args.save_path, f'checkpoint_stream_step_{batch_step}'))
        kge_model.load_state_dict(checkpoint['model_state_dict'])

    logging.info('[BEGIN BATCH EVAL]')
    if args.do_valid:
        logging.info('Evaluating on Currently Known Valid Triples...')
        metrics = kge_model.test_step(kge_model, all_valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)

        if new_valid_triples is not None:
            logging.info('Evaluating on New Valid Dataset...')
            metrics = kge_model.test_step(kge_model, new_valid_triples, all_true_triples, args)
            log_metrics('Valid', step, metrics)

    if args.do_test:
        logging.info('Evaluating on Currently Known Test Triples...')
        metrics = kge_model.test_step(kge_model, all_test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)

        if new_test_triples is not None:
            logging.info('Evaluating on New Test Dataset...')
            metrics = kge_model.test_step(kge_model, new_test_triples, all_true_triples, args)
            log_metrics('Test', step, metrics)

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, new_train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)

    stream_obj = KBStream(args.data_path, stream_init_proportion=args.stream_init_proportion,
                          n_stream_updates=args.n_stream_updates, frac_old_train_samples=args.frac_old_train_samples,
                          sample_nbh=args.sample_nbh, seed=args.stream_seed)

    entity2id, relation2id, known_true_triples, train_triples, valid_triples, test_triples = stream_obj.get_init_kb()
    main_ops(args, entity2id, relation2id, known_true_triples, train_triples,
             valid_triples, None, test_triples, None, 0, len(entity2id), batch_step=0)
    args.valid_steps /= 10

    for batch_ctr, kb_batch_update in enumerate(stream_obj.batch_generator(), start=1):
        entity2id, relation2id, known_true_triples, new_train_triples, all_valid_triples, new_valid_triples, all_test_triples, new_test_triples, nentity_old, nentity_new = kb_batch_update
        main_ops(args, entity2id, relation2id, known_true_triples, new_train_triples,
                 all_valid_triples, new_valid_triples, all_test_triples, new_test_triples, nentity_old, nentity_new,
                 batch_step=batch_ctr)

        
if __name__ == '__main__':
    main(parse_args())
