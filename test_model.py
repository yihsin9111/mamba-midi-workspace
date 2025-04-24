from simba import Text_Mmamba

music_model = Text_Mmamba(
            layers = 5,
            codec_layer = 1,
            vocab_size = 2049,
            d_model = 1024,
            drop_p = 0.3, 
            d_state = 512, 
            num_heads = 8, 
            inner = True, 
            self_atten_layers = [],
            condition_method = 'cross_attention',
            is_pure_mamba = False,
            )

print(music_model.backbone[0].cross_atten.cross_atten.training)