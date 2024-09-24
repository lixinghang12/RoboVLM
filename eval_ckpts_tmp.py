import os

ckpt_paths = [
    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-08/20-24/epoch=2-step=50000.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-08/20-24/2024-07-08_20:25:22.898933-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-09/00-22/epoch=2-step=49999.ckpt', 
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-09/00-22/2024-07-09_00:23:05.880641-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-12/23-45/epoch=2-step=19999.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-12/23-45/2024-07-12_23:46:35.311293-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-12/00-09/epoch=2-step=24999.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-12/00-09/2024-07-12_00:10:07.867182-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-12/22-50/epoch=3-step=27498.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-12/22-50/2024-07-12_22:52:34.169598-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-16/14-09/epoch=3-step=27498.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-16/14-09/2024-07-16_14:10:14.960742-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-16/15-31/epoch=2-step=35000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-16/15-31/2024-07-16_15:32:47.575957-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/qwen/calvin_finetune/2024-07-19/16-57/epoch=2-step=62499.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/qwen/calvin_finetune/2024-07-19/16-57/2024-07-19_16:58:38.185369-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/qwen/calvin_finetune/2024-07-19/16-57/epoch=3-step=82498.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/qwen/calvin_finetune/2024-07-19/16-57/2024-07-19_16:58:38.185369-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/qwen/calvin_finetune/2024-07-21/01-03/epoch=4-step=34997.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/qwen/calvin_finetune/2024-07-21/01-03/2024-07-21_01:04:16.505673-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/qwen/calvin_finetune/2024-07-22/17-05/epoch=1-step=12499.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/qwen/calvin_finetune/2024-07-22/17-05/2024-07-22_17:05:54.396891-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-24/16-53/epoch=0-step=12500.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-24/16-53/2024-07-24_16:55:20.507641-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-23/00-21/epoch=1-step=34999.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-23/00-21/2024-07-23_00:22:43.604769-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/qwen/calvin_finetune/2024-07-22/12-17/epoch=2-step=39999.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/qwen/calvin_finetune/2024-07-22/12-17/2024-07-22_12:18:02.509362-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/qwen/calvin_finetune/2024-07-22/17-05/epoch=3-step=34999.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/qwen/calvin_finetune/2024-07-22/17-05/2024-07-22_17:05:54.396891-project.json'),

    ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-26/01-09/epoch=3-step=35000.pt',
    '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-26/01-09/2024-07-26_01:10:12.903139-project.json')
]

for i, (ckpt, config) in enumerate(ckpt_paths):
    os.system('bash run_llava_eval_raw_ddp_torchrun.sh {} {}'.format(ckpt, config))