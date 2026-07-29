# Pretrained Policy

Place the pretrained NavDP policy checkpoint in this directory before training
or evaluation. The default training configuration expects:

```text
pretrain_model/navdp_pretrained.ckpt
```

Alternatively, pass a checkpoint explicitly with `train.py --pretrained_model`
or `eval/scripts/start_policy_server.sh --checkpoint`.
