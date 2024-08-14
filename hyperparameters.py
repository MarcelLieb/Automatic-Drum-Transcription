from main import main as train
from settings import Config

fast_decent_cnn = {
    'learning_rate': 0.0005060652326765156,
    'epochs': 20,
    'batch_size': 1024,
    'weight_decay': 2.7219485506860926e-05,
    'positive_weight': 1.0,
    'ema': False,
    'scheduler': False,
    'early_stopping': "None",
    'dataset_version': 'L',
    'splits': "(0.85, 0.15, 0.0)",
    'num_workers': 64,
    'min_save_score': 0.46,
    'test_batch_size': 1,
    'train_set': 'a2md_train',
    'model_settings': 'cnn',
    'peak_mean_range': 2,
    'peak_max_range': 2,
    'onset_cooldown': 0.021,
    'detect_tolerance': 0.025,
    'ignore_beats': True,
    'min_test_score': 0.48,
    'pr_points': 1000,
    'sample_rate': 44100,
    'hop_size': 441,
    'fft_size': 2048,
    'n_mels': 96,
    'center': True,
    'pad_mode': 'constant',
    'mel_min': 20.0,
    'mel_max': 20000.0,
    'normalize': False,
    'mapping': "Three class standard",
    'pad_annotations': True,
    'pad_value': 0.5,
    'time_shift': 0.015,
    'beats': False,
    'segment_type': 'frame',
    'frame_length': 0.2035858401526392,
    'frame_overlap': 0.1,
    'label_lead_in': 0.25,
    'label_lead_out': 0.1,
    'num_channels': 35,
    'num_residual_blocks': 0,
    'num_feature_layers': 3,
    'channel_multiplication': 2,
    'dropout': 0.1537819399758399,
    'causal': True,
    'flux': False,
    'activation': 'SELU',
    'classifier_dim': 174,
    'down_sample_factor': 4
}


def main():
    config = Config.from_flat_dict(fast_decent_cnn)
    train(config)


if __name__ == '__main__':
    main()
