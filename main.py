import yaml
from pathlib import Path
from src.data_preparation import DataPreparator
from src.model import PosterDetectionModel
from src.trainer import ModelTrainer

def main():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    data_prep = DataPreparator(config)
    model = PosterDetectionModel(config)
    trainer = ModelTrainer(model, config)

    # Prepare data
    train_dataset, val_dataset = data_prep.prepare_dataset()

    # Train model
    trainer.train(train_dataset, val_dataset)

    # Save results
    trainer.save_results()

if __name__ == "__main__":
    main()