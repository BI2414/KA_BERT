import argparse


def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data/wjh/graduate/AugData/cMedQA2")
    parser.add_argument("--model_name", type=str, default="data/wjh/graduate/data/bert-base-chinese")
    parser.add_argument("--save_dir", type=int, default="data/wjh/graduate/data/save")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=int, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=128)

    parser.add_argument("--embed_dim", type=str, default=128)
    parser.add_argument("--share_low_layers", type=str, default=3)
    parser.add_argument("--use_gate", type=int, default=True)
    parser.add_argument("--kl_weight", type=int, default=0.1)
    parser.add_argument("--use_cross_attn", type=int, default=True)
    parser.add_argument("--contrastive_margin", type=str, default=0.2)
    parser.add_argument("--adam_epsilon", type=str, default=1e-8)
    parser.add_argument("--weight_decay", type=int, default=0.01)
    return parser