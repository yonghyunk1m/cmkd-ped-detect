import argparse
import yaml
import pytorch_lightning as pl

from data.datamodule import ASPEDDataModule
from models.seq2seq import ASPEDLightningModel
from data.utils import get_train_test_splits

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Evaluate the ASPED Pedestrian Counting Model")
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the YAML configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the saved .ckpt model file')
    parser.add_argument('--test_session', type=str, required=True, 
                        help='Session held out for testing (e.g., Session_02152024)')
    parser.add_argument('--feature_dir', type=str, default=None,
                        help='Override feature root directory')
    args = parser.parse_args()

    # 1. Load Configurations
    config = load_config(args.config)
    task_type = config['model']['task']
    n_classes = config['model'].get('n_classes', 1)
    
    print(f"🚀 Initializing Evaluation for task: {task_type.upper()} (Classes: {n_classes})")
    print(f"📂 Evaluating on Hold-out Session: {args.test_session}")
    print(f"📦 Loading weights from: {args.checkpoint}")

    # 2. Setup DataModule for Testing
    feature_dir = (
        args.feature_dir
        or config.get('data_params', {}).get('feature_root_dir')
        or "/path/to/ASPED_v.c"
    )
    train_val_list, test_list = get_train_test_splits(
        feature_root_dir=feature_dir, test_session=args.test_session, window_size_sec=10
    )
    
    datamodule = ASPEDDataModule(
        train_val_data_list=train_val_list,
        test_data_list=test_list,
        batch_size=config['dataloader_params']['batch_size'],
        task_type=task_type,
        num_workers=config['dataloader_params']['num_workers'],
        n_classes=n_classes 
    )

    # 3. Load the Model from Checkpoint
    model = ASPEDLightningModel.load_from_checkpoint(
        checkpoint_path=args.checkpoint
    )
    
    # Ensure model is in evaluation mode
    model.eval()

    # 4. Initialize Trainer in Test Mode
    trainer = pl.Trainer(
        accelerator=config['trainer']['args']['accelerator'],
        devices=1,
        logger=False # Disable logging for a clean test output in console
    )

    # 5. Execute Testing
    print(f"\n🔍 Running test suite on {len(test_list)} hold-out samples...")
    trainer.test(model, datamodule=datamodule)
    
    print("\n✅ Evaluation Complete!")
    
if __name__ == "__main__":
    main()