from torch.utils.data import Dataset
import pandas as pd
import torch


class SpectrumDataset(Dataset):
    """Dataset loader for spectral data with train/test split capability.
    
    Attributes:
        DATA_DIMENSION (int): Dimension length of input feature vectors
        mode (str): Dataset mode ('train'/'test')
        random_state (int): Random seed
        data_df (pd.DataFrame): Loaded dataset
    """
    
    DATA_DIMENSION = 8000  # Input feature dimension length

    def __init__(self, mode: str, random_state: int = 1) -> None:
        """Initialize the dataset loader.
        
        Args:
            mode: Dataset mode ('train'/'test')
            random_state: Random seed (default: 1)
        """
        super().__init__()
        self.mode = mode.lower()
        self.random_state = random_state
        self.data_df = self._load_data_from_csv('../IR_Raman_azo.csv')

    def __getitem__(self, index):
        """Get a single data sample.
        
        Returns:
            tuple: (spectral_features, target_value, molecule_id, 
                   frequency_id, frequency_value, molecule_type)
        """
        features = torch.tensor(
            self.data_df.iloc[index, :self.DATA_DIMENSION], 
            dtype=torch.float
        )
        target = torch.tensor(
            [float(self.data_df.iloc[index, self.DATA_DIMENSION + 5])]
        )
        molecule_id = torch.tensor(
            [float(self.data_df.iloc[index, self.DATA_DIMENSION + 1])]
        )
        frequency_id = torch.tensor(
            [float(self.data_df.iloc[index, self.DATA_DIMENSION + 2])]
        )
        frequency_value = torch.tensor(
            [float(self.data_df.iloc[index, self.DATA_DIMENSION + 3])]
        )
        molecule_type = torch.tensor(
            [float(self.data_df.iloc[index, self.DATA_DIMENSION])]
        )
        
        # Reshape features (4000x2 -> 2x4000)
        features = features.reshape(int(self.DATA_DIMENSION / 2), 2).T
        
        return (features, target, molecule_id, 
                frequency_id, frequency_value, molecule_type)

    def __len__(self):
        return len(self.data_df)

    def _load_data_from_csv(self, file_path: str) -> pd.DataFrame:
        """Load dataset from CSV file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Processed training or test set
            
        Raises:
            ValueError: When invalid mode is provided
        """
        raw_df = pd.read_csv(file_path, header=None)
        shuffled_df = raw_df.sample(
            frac=1, 
            random_state=self.random_state
        ).reset_index(drop=True)

        if self.mode == 'train':
            return shuffled_df.iloc[0:10]
        elif self.mode == 'test':
            return shuffled_df.iloc[10:20]
        else:
            raise ValueError(
                f"Mode must be either 'train' or 'test', got '{self.mode}'"
            )


if __name__ == '__main__':
    # Example usage
    dataset = SpectrumDataset('train')
    sample = dataset[0]

