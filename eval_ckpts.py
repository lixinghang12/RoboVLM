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

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-26/01-09/epoch=3-step=35000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-26/01-09/2024-07-26_01:10:12.903139-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-26/15-09/epoch=3-step=40000.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-26/15-09/2024-07-26_15:10:01.652163-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-29/18-02/epoch=2-step=25000.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-29/18-02/2024-07-29_18:02:45.267097-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-31/21-44/epoch=1-step=10000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-31/21-44/2024-07-31_21:46:02.505101-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-31/11-57/epoch=1-step=15000.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-31/11-57/2024-07-31_11:57:51.901957-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-03/06-28/epoch=3-step=30000.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-03/06-28/2024-08-03_06:28:54.394895-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-03/06-28/epoch=2-step=25000-v1.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-03/06-28/2024-08-03_06:29:35.228922-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-31/22-03/epoch=2-step=30000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-31/22-03/2024-07-31_22:03:59.754788-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-05/14-57/epoch=0-step=15000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-05/14-57/2024-08-05_14:58:53.893842-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-05/15-04/epoch=1-step=19999.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-05/15-04/2024-08-05_15:05:41.144263-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-31/00-13/epoch=2-step=49999.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-31/00-13/2024-07-31_00:14:37.815947-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-07-31/00-13/epoch=0-step=10000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-07-31/00-13/2024-07-31_00:14:37.815947-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-06/17-33/epoch=1-step=19999.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-06/17-33/2024-08-06_17:33:38.927235-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-08/11-59/epoch=0-step=10000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-08/11-59/2024-08-08_11:59:59.704237-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-08/18-20/epoch=0-step=5000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-08/18-20/2024-08-08_18:21:06.780836-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-08/19-46/epoch=0-step=5000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-08/19-46/2024-08-08_19:46:47.003411-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-09/10-22/epoch=4-step=104998.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-09/10-22/2024-08-09_10:23:01.353657-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-09/10-22/epoch=0-step=20000-v1.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-09/10-22/2024-08-09_10:23:19.623344-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-14/19-46/epoch=0-step=20000.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-14/19-46/2024-08-14_19:47:00.143718-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-15/00-09/epoch=0-step=15000.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-15/00-09/2024-08-15_00:10:39.324546-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-15/19-30/epoch=0-step=20000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-15/19-30/2024-08-15_19:31:30.288271-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-19/17-01/epoch=0-step=2500.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-19/17-01/2024-08-19_17:01:51.967601-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-19/20-00/epoch=0-step=7500.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-19/20-00/2024-08-19_20:01:38.241545-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-19/20-00/epoch=0-step=7500-v1.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-19/20-00/2024-08-19_20:01:40.439332-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-19/20-25/epoch=0-step=4000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-19/20-25/2024-08-19_20:26:31.373527-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-19/20-06/epoch=9-step=20000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-19/20-06/2024-08-19_20:07:13.576109-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-19/20-00/epoch=0-step=10000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-19/20-00/2024-08-19_20:01:38.241545-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-19/20-00/epoch=0-step=10000-v1.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-19/20-00/2024-08-19_20:01:40.439332-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-20/17-16/epoch=0-step=10000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-20/17-16/2024-08-20_17:16:48.929673-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-20/15-30/epoch=0-step=10000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-20/15-30/2024-08-20_15:30:58.729942-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-20/17-16/epoch=0-step=30000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-20/17-16/2024-08-20_17:16:48.929673-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-19/21-29/epoch=1-step=7999.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-19/21-29/2024-08-19_21:30:02.750694-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/12-00/epoch=0-step=5000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/12-00/2024-08-21_12:01:15.162672-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/12-00/epoch=0-step=10000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/12-00/2024-08-21_12:01:15.162672-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/19-25/epoch=0-step=22500.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/19-25/2024-08-21_19:25:42.813239-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/19-25/epoch=0-step=27500.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/19-25/2024-08-21_19:25:42.813239-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/19-25/epoch=0-step=30000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/19-25/2024-08-21_19:25:42.813239-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/19-25/epoch=0-step=35000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/19-25/2024-08-21_19:25:42.813239-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/19-25/epoch=1-step=47500.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/19-25/2024-08-21_19:25:42.813239-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/18-16/epoch=1-step=60000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/18-16/2024-08-21_18:17:07.064426-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/epoch=0-step=40000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/2024-08-23_00:57:15.564401-project.json')

    #  ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/19-25/epoch=1-step=67500.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/19-25/2024-08-21_19:25:42.813239-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-22/16-04/epoch=1-step=40000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-22/16-04/2024-08-22_16:04:56.207556-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/epoch=1-step=80000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/2024-08-23_00:57:15.564401-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/18-16/epoch=2-step=110000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/18-16/2024-08-21_18:17:07.064426-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/19-25/epoch=2-step=97500.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-21/19-25/2024-08-21_19:25:42.813239-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/qwen/calvin_finetune/2024-08-21/21-57/epoch=0-step=40000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/qwen/calvin_finetune/2024-08-21/21-57/2024-08-21_21:58:08.677187-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-57/epoch=2-step=150000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-57/2024-08-23_00:58:29.036489-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/epoch=2-step=160000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/2024-08-23_00:57:15.564401-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/epoch=2-step=170000.pt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/2024-08-23_00:57:15.564401-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-57/epoch=2-step=170000.pt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-57/2024-08-23_00:58:29.036489-project.json')

    #  ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/epoch=2-step=180000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/2024-08-23_00:57:15.564401-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-46/epoch=0-step=10000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-46/2024-08-25_17:47:19.667561-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/06-12/epoch=0-step=40000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/06-12/2024-08-25_06:13:31.555596-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/epoch=2-step=200000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/2024-08-23_00:57:15.564401-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/epoch=1-step=140000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/2024-08-23_00:57:15.564401-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/qwen/calvin_finetune/2024-08-25/11-53/epoch=0-step=30000.pt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/qwen/calvin_finetune/2024-08-25/11-53/2024-08-25_11:54:19.698835-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-46/epoch=0-step=30000.pt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-46/2024-08-25_17:47:19.667561-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-43/epoch=0-step=25000.pt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-43/2024-08-25_17:44:23.789007-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-46/epoch=0-step=40000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-46/2024-08-25_17:47:19.667561-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/epoch=3-step=230000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/2024-08-23_00:57:15.564401-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-46/epoch=0-step=50000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-46/2024-08-25_17:47:19.667561-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/epoch=3-step=280000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/2024-08-23_00:57:15.564401-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-43/epoch=1-step=60000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-43/2024-08-25_17:44:23.789007-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-26/14-23/epoch=0-step=35000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-26/14-23/2024-08-26_14:24:01.136627-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-26/14-14/epoch=0-step=70000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-26/14-14/2024-08-26_14:15:16.584359-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-46/epoch=1-step=90000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-46/2024-08-25_17:47:19.667561-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/qwen/calvin_finetune/2024-08-25/11-53/epoch=1-step=60000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/qwen/calvin_finetune/2024-08-25/11-53/2024-08-25_11:54:19.698835-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/06-12/epoch=2-step=170000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/06-12/2024-08-25_06:13:31.555596-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/epoch=4-step=330000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/2024-08-23_00:57:15.564401-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-46/epoch=1-step=120000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-46/2024-08-25_17:47:19.667561-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-43/epoch=2-step=90000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-43/2024-08-25_17:44:23.789007-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-26/14-23/epoch=1-step=65000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-26/14-23/2024-08-26_14:24:01.136627-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/qwen/calvin_finetune/2024-08-27/20-07/epoch=6-step=30000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/qwen/calvin_finetune/2024-08-27/20-07/2024-08-27_20:07:37.965004-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/qwen/calvin_finetune/2024-08-25/11-53/epoch=2-step=80000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/qwen/calvin_finetune/2024-08-25/11-53/2024-08-25_11:54:19.698835-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-26/14-14/epoch=1-step=140000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-26/14-14/2024-08-26_14:15:16.584359-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/epoch=4-step=350000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/2024-08-23_00:57:15.564401-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-46/epoch=2-step=150000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-46/2024-08-25_17:47:19.667561-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/06-12/epoch=3-step=230000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/06-12/2024-08-25_06:13:31.555596-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/epoch=4-step=370000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/2024-08-23_00:57:15.564401-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-26/14-14/epoch=2-step=190000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-26/14-14/2024-08-26_14:15:16.584359-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-46/epoch=2-step=170000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-46/2024-08-25_17:47:19.667561-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/qwen/calvin_finetune/2024-08-25/11-53/epoch=2-step=110000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/qwen/calvin_finetune/2024-08-25/11-53/2024-08-25_11:54:19.698835-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-27/04-12/epoch=2-step=80000.pt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-27/04-12/2024-08-27_04:12:51.538520-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/06-12/epoch=4-step=300000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/06-12/2024-08-25_06:13:31.555596-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-43/epoch=2-step=110000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-25/17-43/2024-08-25_17:44:23.789007-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-27/04-12/epoch=2-step=95000-v1.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-27/04-12/2024-08-27_04:12:58.162683-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-29/22-36/epoch=0-step=15000.pt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-29/22-36/2024-08-29_22:37:21.360171-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-29/22-37/epoch=0-step=10000.pt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-29/22-37/2024-08-29_22:37:52.271845-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/epoch=3-step=230000.pt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-23/00-56/2024-08-23_00:57:15.564401-project.json')

    #  ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-29/22-37/epoch=0-step=15000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-29/22-37/2024-08-29_22:37:52.271845-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-29/22-36/epoch=1-step=55000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-29/22-36/2024-08-29_22:37:21.360171-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-28/17-31/epoch=1-step=70000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-28/17-31/2024-08-28_17:32:10.400486-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-28/16-39/epoch=2-step=210000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-28/16-39/2024-08-28_16:39:42.422072-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-29/22-37/epoch=1-step=24999.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-29/22-37/2024-08-29_22:37:52.271845-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/qwen/calvin_finetune/2024-08-28/18-03/epoch=0-step=60000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/qwen/calvin_finetune/2024-08-28/18-03/2024-08-28_18:04:08.540478-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-08-31/21-46/epoch=2-step=69060.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs-v2/video_pretrain_manipulation/calvin_finetune/2024-08-31/21-46/2024-08-31_21:46:40.148568-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-31/12-55/epoch=1-step=4999.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-31/12-55/2024-08-31_12:56:22.831211-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-31/12-55/epoch=4-step=19999.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-31/12-55/2024-08-31_12:56:22.831211-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-28/17-31/epoch=2-step=100000.pt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-28/17-31/2024-08-28_17:32:10.400486-project.json'),
   
    # garbage
    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-31/11-18/epoch=1-step=7499.pt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-31/11-18/2024-08-31_11:19:30.203670-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-31/12-55/epoch=2-step=12499.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-31/12-55/2024-08-31_12:56:22.831211-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-08-31/21-46/epoch=3-step=92080.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs-v2/video_pretrain_manipulation/calvin_finetune/2024-08-31/21-46/2024-08-31_21:46:40.148568-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-28/16-39/epoch=4-step=320000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-28/16-39/2024-08-28_16:39:42.422072-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-09-01/23-50/epoch=1-step=46040.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/calvin_finetune/2024-09-01/23-50/2024-09-01_23:50:58.321390-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-09-01/22-57/epoch=1-step=4999.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-09-01/22-57/2024-09-01_22:59:20.251439-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-09-02/13-29/epoch=0-step=23020.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/calvin_finetune/2024-09-01/23-50/2024-09-01_23:50:58.321390-project.json'),

    # ('/mnt/bn/robotics-data-hl/lxh/robot-flamingo/RobotFlamingoDBG_ABC_D/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/calvin_finetune/2024-09-01/23-50/2024-09-01_23:50:58.321390-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-31/11-18/epoch=3-step=14999.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-31/11-18/2024-08-31_11:19:30.203670-project.json')

    # ('/mnt/bn/robotics-data-hl/lxh/robot-flamingo/RobotFlamingoDBG_ABC_D/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/calvin_finetune/2024-09-01/23-50/2024-09-01_23:50:58.321390-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-29/22-36/epoch=3-step=145000.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-29/22-36/2024-08-29_22:37:21.360171-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-08-29/22-37/epoch=3-step=69998.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-08-29/22-37/2024-08-29_22:37:52.271845-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-09-03/01-02/epoch=0-step=17860.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/calvin_finetune/2024-09-03/01-02/2024-09-03_01:03:27.893914-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-09-03/01-02/epoch=1-step=35720.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/calvin_finetune/2024-09-03/01-02/2024-09-03_01:03:27.893914-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-09-03/16-36/epoch=0-step=23020.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/calvin_finetune/2024-09-03/16-36/2024-09-03_16:36:50.757615-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-09-02/23-36/epoch=0-step=4375.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-09-02/23-36/2024-09-02_23:37:43.465907-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-09-03/00-11/epoch=1-step=5000.pt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-09-03/00-11/2024-09-03_00:12:25.741277-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-09-03/23-39/epoch=0-step=23020.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/calvin_finetune/2024-09-03/23-39/2024-09-03_23:40:12.805385-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-09-03/23-39/epoch=1-step=46040.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/calvin_finetune/2024-09-03/23-39/2024-09-03_23:40:12.805385-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-09-03/23-39/epoch=2-step=69060.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/calvin_finetune/2024-09-03/23-39/2024-09-03_23:40:12.805385-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-09-04/00-58/epoch=1-step=4999.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-09-04/00-58/2024-09-04_00:59:29.559378-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-09-04/15-27/epoch=1-step=5302.pt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-09-04/15-27/2024-09-04_15:28:04.678542-project.json')

    # ('/mnt/bn/robotics-data-hl/lxh/robot-flamingo/RobotFlamingoDBGABCD/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_0.pth',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/calvin_finetune/2024-09-03/23-39/2024-09-03_23:40:12.805385-project.json'),

    # ('/mnt/bn/robotics-data-hl/lxh/robot-flamingo/RobotFlamingoDBGABCD/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_1.pth',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/calvin_finetune/2024-09-03/23-39/2024-09-03_23:40:12.805385-project.json')

    # ('/mnt/bn/robotics-data-hl/lxh/robot-flamingo/RobotFlamingoDBGABCD/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/calvin_finetune/2024-09-03/23-39/2024-09-03_23:40:12.805385-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-09-04/23-41/epoch=2-step=69060.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/calvin_finetune/2024-09-04/23-41/2024-09-04_23:42:25.125926-project.json'),

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-09-04/23-44/epoch=3-step=23020.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/calvin_finetune/2024-09-04/23-44/2024-09-04_23:45:27.705512-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-06-24/default/epoch=2-step=71931.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/calvin_finetune/2024-06-24/default/2024-06-24_16:39:20.620112-project.json')

    # ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-09-04/23-44/epoch=4-step=28775.ckpt',
    # '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/calvin_finetune/2024-09-04/23-44/2024-09-04_23:45:27.705512-project.json')

    ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-09-23/18-28/epoch=0-step=18701.ckpt',
    '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-09-23/18-28/2024-09-23_18:29:18.980334-project.json'),

    ('/mnt/bn/robotics-data-lxh-lq-v2/checkpoints/video_pretrain_manipulation/llava/calvin_finetune/2024-09-23/20-24/epoch=0-step=18701.ckpt',
    '/mnt/bn/robotics-data-lxh-lq-v2/logs/video_pretrain_manipulation/llava/calvin_finetune/2024-09-23/20-24/2024-09-23_20:25:05.802920-project.json')
]   

for i, (ckpt, config) in enumerate(ckpt_paths):
    print("evaluating checkpoint {}".format(ckpt))
    os.system('bash run_llava_eval_raw_ddp_torchrun.sh {} {}'.format(ckpt, config))