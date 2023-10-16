import argparse
import datetime
import json
import logging
import os
import shutil

from utils.util import parse_time


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', default=True, help='use GPU')

    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")

    parser.add_argument('--data_path', type=str, default=None, help="KG data path")
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int,
                        help="negative entities sampled per query")
    parser.add_argument('-d', '--hidden_dim', default=400, type=int, help="embedding dimension")
    parser.add_argument('-g', '--gamma', default=12.0, type=float, help="margin in the loss")
    parser.add_argument('-b', '--batch_size', default=512, type=int, help="batch size of queries")
    parser.add_argument('--test_batch_size', default=500, type=int, help='valid/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int, help="used to speed up torch.dataloader")
    parser.add_argument('-save', '--save_path', default=None, type=str,
                        help="no need to set manually, will configure automatically")
    parser.add_argument('--max_steps', default=100000, type=int, help="maximum iterations to train")

    parser.add_argument('--warm_up_steps', default=20000, type=int)

    parser.add_argument('--valid_steps', default=5000, type=int, help="evaluate validation queries every xx steps")
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--num_ents', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--num_rels', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--geo', default='vec', type=str,
                        help='the reasoning model, vec for GQE, box for Query2box, beta for BetaE')

    parser.add_argument('--tasks', default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', type=str,
                        help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=3407, type=int, help="random seed")
    parser.add_argument('--prefix', default=None, type=str, help='prefix of the log path')
    parser.add_argument('--ckpt_path', default=None, type=str, help='path for loading the checkpoints')
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'],
                        help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument('--fp16', action='store_true')

    parser.add_argument('--label_smoothing', default=0.1, type=float)
    parser.add_argument('-de', '--dim_ent_embedding', default=None, type=int)
    parser.add_argument('-dr', '--dim_rel_embedding', default=None, type=int)

    parser.add_argument('--kge_ckpt_path', default=None, type=str)
    parser.add_argument('--not_freeze_kge', default=False, action='store_true')

    # self-attention settings
    parser.add_argument('--num_attention_heads', default=10, type=int)
    parser.add_argument('--hidden_dropout_prob', default=0.1, type=float)
    parser.add_argument('--intermediate_size', default=1000, type=int)
    parser.add_argument('--layer_norm_eps', default=1e-12, type=float)
    parser.add_argument('--num_hidden_layers', default=4, type=int)
    parser.add_argument('--hidden_size', default=500, type=int)

    parser.add_argument('--token_embeddings', default='0', type=str,
                        help='1: type_embeds, 2: layer_embeds, 3: op_embeds')
    parser.add_argument('--enc_dist', choices=['u', 'd', 'n', 'no'], default='d',
                        help='u: undirected dist, d: directed dist, n: only neighbors')
    parser.add_argument('--ignore_1p', action='store_true')

    args = parser.parse_args(args)
    args = post_init_args(args)

    return args


def post_init_args(args):
    """
    post init args, set save_path and so on.
    """

    tasks = args.tasks.split('.')
    ori_task = args.tasks
    args.tasks = tasks
    for task in tasks:
        if 'n' in task and args.geo in ['box', 'vec']:
            assert False, "Q2B and GQE cannot handle queries with negation"
    if args.evaluate_union == 'DM':
        assert args.geo == 'beta', "only BetaE supports modeling union using De Morgan's Laws"

    cur_time = parse_time()
    prefix = 'logs' if args.prefix is None else args.prefix
    args.save_path = os.path.join(prefix, args.data_path.split('/')[-1], ori_task, args.geo)

    if args.ckpt_path is not None:
        args.save_path = args.ckpt_path

        config_path = os.path.join(args.ckpt_path, 'config.json')
        print(f'Loading config from {config_path}...')
        with open(config_path, 'r') as f:
            config = json.load(f)
        args.hidden_dim = config['hidden_dim']
        args.gamma = config['gamma']
    else:
        if args.geo in ['box']:
            tmp_str = "g-{}-mode-{}".format(args.gamma, args.box_mode)
        elif args.geo in ['vec']:
            tmp_str = "g-{}".format(args.gamma)
        elif args.geo == 'beta':
            tmp_str = "g-{}-mode-{}".format(args.gamma, args.beta_mode)
        else:
            tmp_str = args.geo

        args.save_path = os.path.join(args.save_path, tmp_str, cur_time)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.do_train:
        # save key file
        shutil.copy('./main.py', os.path.join(args.save_path, 'main.py'))
        shutil.copy('./query_format_converter.py', os.path.join(args.save_path, 'query_format_converter.py'))
        shutil.copy('Models/query2triple.py', os.path.join(args.save_path, 'prompt_cqa.py'))
        shutil.copy('./Models/modeling_bert.py', os.path.join(args.save_path, 'modeling_bert.py'))
        shutil.copy('./Models/transformer_conv.py', os.path.join(args.save_path, 'transformer_conv.py'))
        shutil.copy('./utils/util.py', os.path.join(args.save_path, 'util.py'))

    # set logger after setting save_path
    set_logger(args)

    logging.info(f'logging to {args.save_path}')

    # set num_ents and num_rels
    with open('%s/stats.txt' % args.data_path) as f:
        entrel = f.readlines()
        num_ents = int(entrel[0].split(' ')[-1])
        num_rels = int(entrel[1].split(' ')[-1])

    # contain inverse rel already
    args.num_ents = num_ents
    args.num_rels = num_rels

    if args.do_train:
        with open(args.save_path + '/config.json', 'w') as f:
            json.dump(vars(args), f)

    return args


def set_logger(args):
    '''
    Write logs to console and log file
    '''
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    # def beijing(sec, what):
    #     beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    #     return beijing_time.timetuple()
    #
    # logging.Formatter.converter = beijing

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
