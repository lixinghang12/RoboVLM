from model.backbone.flamingo_video import RoboFlamingoVideo
from train.flamingo_trainer import FlamingoTrainer


class FlamingoVideoTrainer(FlamingoTrainer):
    def __init__(self, configs):
        super(FlamingoVideoTrainer, self).__init__(configs)

    def _init_policy(self):

        model = RoboFlamingoVideo(
            vision_encoder_configs=self.configs['vision_encoder'],
            tokenizer_configs=self.configs['tokenizer'],
            train_setup_configs=self.configs['train_setup'],
            fwd_head_configs=None,
            llm_configs=self.configs['llm'],
            window_size=self.configs['window_size'],
            use_hand_rgb=self.use_hand_rgb,
            act_head_configs=self.configs['act_head'],
            fwd_pred_next_n=self.configs['fwd_pred_next_n']
        )

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._main_rank_print(f"Flamingo-Video Model Parameters: {total_params / 1000000:.2f}M")
        return model

    def _process_batch(self, batch):
        """
        Action Prediction:
            args: rgb, language, attention_mask, hand_rgb, action
            reformat: action to input and target (seq_len = window size + chunck size)
        Video Prediction:
            args: rgb, language, attention mask, hand_rgb
            reformat: rgb, [hand_rgb] to input and target (seq_len = window size + chunck size)
        Video Caption:
            args: rgb, language, attention_mask
            reformat: Identity
        Image Caption:
            args: rgb, language, attention_mask
            reformat: Identity
            seq_len = 1
        """
        # print(type(batch), len(batch))
        if isinstance(batch, list):
            # print(batch[0].keys())
            batch = batch[0]
        rgb = batch['rgb'].cuda()
        if len(rgb.shape) == 4:
            rgb = rgb.unsqueeze(1)
        assert len(rgb.shape) == 5
        seq_len = self.configs['window_size']
        if isinstance(batch['text'], list) and isinstance(batch['text'][0], str):
            raise ValueError('The raw text data is not supported')
        else:
            language = batch['text'].cuda()
            text_mask = batch['text_mask'].cuda()
        
        if batch.get('action', None) is not None:
            action = batch['action'].cuda()
        else:
            action = None

        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = batch['attention_mask'].cuda()
        
        if self.use_hand_rgb and batch.get('hand_rgb', None) is not None:
            hand_rgb = batch['hand_rgb'].cuda()
        else:
            hand_rgb = None

        # Split arm and gripper action
        arm_action = None
        gripper_action = None

        if action is not None:
            arm_action = action[:, :, :6]  # b,len,act_dim-1
            gripper_action = action[:, :, 6]  # b,len
            gripper_action = gripper_action.long()

        fwd_rgb_chunck = batch.get('fwd_rgb_chunck', None)
        fwd_hand_rgb_chunck = batch.get('fwd_hand_rgb_chunck', None)
        if fwd_rgb_chunck is not None:
            fwd_rgb_chunck = fwd_rgb_chunck.cuda()
        if fwd_hand_rgb_chunck is not None:
            fwd_hand_rgb_chunck = fwd_hand_rgb_chunck.cuda()

        arm_action_chunck = None
        gripper_action_chunck = None
        action_chunck = batch.get('action_chunck', None)
        if action_chunck is not None:
            action_chunck = action_chunck.cuda()
            arm_action_chunck = action_chunck[..., :6]
            gripper_action_chunck = action_chunck[..., -1]
        
        rgb = rgb[:, :seq_len]
        if hand_rgb is not None:
            hand_rgb = hand_rgb[:, :seq_len]
        
        chunck_mask = batch.get('chunck_mask', None)
        if chunck_mask is not None:
            chunck_mask = chunck_mask.cuda()
        
        # data preparation for discrete action inputs and labels
        instr_and_action_ids = batch.get("instr_and_action_ids", None)
        if instr_and_action_ids is not None:
            instr_and_action_ids = instr_and_action_ids.cuda()
        
        instr_and_action_labels = batch.get("instr_and_action_labels", None)
        if instr_and_action_labels is not None:
            instr_and_action_labels = instr_and_action_labels.cuda()
        
        instr_and_action_mask = batch.get("instr_and_action_mask", None)
        if instr_and_action_mask is not None:
            instr_and_action_mask = instr_and_action_mask.cuda()

        return rgb, hand_rgb, attention_mask, language, text_mask, fwd_rgb_chunck, fwd_hand_rgb_chunck,\
        arm_action, gripper_action, arm_action_chunck, gripper_action_chunck, chunck_mask, instr_and_action_ids, instr_and_action_labels, instr_and_action_mask