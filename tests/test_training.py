import pytest
import torch
import torch.nn as nn
from src.training.losses import PinballLoss, ONIConditionedAsymmetricLoss, CombinedLoss
from src.training.trainer import Trainer

def test_pinball_loss():
    """Verify Pinball Loss correctly penalizes different quantiles."""
    loss_fn = PinballLoss(quantiles=[0.1, 0.5, 0.9])
    
    # Shape: (batch_size, num_targets, num_quantiles)
    # Predicting 100 for all quantiles
    y_pred = torch.ones(2, 1, 3) * 100.0
    
    # True value is 150 (under-prediction)
    y_true = torch.ones(2, 1) * 150.0
    
    loss = loss_fn(y_pred, y_true)
    assert loss.item() > 0

    # True value is 50 (over-prediction)
    y_true_over = torch.ones(2, 1) * 50.0
    loss_over = loss_fn(y_pred, y_true_over)
    assert loss_over.item() > 0

def test_oni_asymmetric_loss():
    """Verify asymmetric loss penalizes overestimation during high ONI (El Nino)."""
    loss_fn = ONIConditionedAsymmetricLoss(threshold=0.5, scaling_factor=2.0)
    
    # Predict 150, True 100 (Overestimating inflow)
    y_pred = torch.ones(2, 1) * 150.0
    y_true = torch.ones(2, 1) * 100.0
    
    # Neutral ONI (0.0) -> Loss should be 0 because gamma = 0
    oni_neutral = torch.zeros(2)
    loss_neutral = loss_fn(y_pred, y_true, oni_neutral)
    assert loss_neutral.item() == 0.0
    
    # El Nino ONI (1.5) -> Loss should be positive
    oni_nino = torch.ones(2) * 1.5
    loss_nino = loss_fn(y_pred, y_true, oni_nino)
    assert loss_nino.item() > 0.0

def test_combined_loss():
    """Verify combined loss integrates both pinball and asymmetric penalty."""
    loss_fn = CombinedLoss()
    
    y_pred = torch.ones(2, 1, 3) * 150.0
    y_true = torch.ones(2, 1) * 100.0
    oni = torch.ones(2) * 1.5
    
    loss = loss_fn(y_pred, y_true, oni)
    assert loss.item() > 0.0

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3) # 3 quantiles
        
    def forward(self, node_features, climate_indices, edge_index):
        # Ignore graph/climate structure, just mock an output
        batch_size = node_features.size(0)
        out = self.fc(node_features)
        return out.unsqueeze(1) # Add targets dimension back

class MockDataLoader:
    def __init__(self):
        self.data = [
            {
                'node_features': torch.randn(4, 10),
                'climate_indices': torch.randn(4, 5),
                'edge_index': torch.zeros(2, 0, dtype=torch.long),
                'targets': torch.randn(4, 1),
                'oni': torch.randn(4)
            }
        ]
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)

def test_trainer_epoch():
    """Verify Trainer can execute a single forward/backward pass epoch."""
    model = MockModel()
    criterion = CombinedLoss()
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    trainer = Trainer(model=model, criterion=criterion, edge_index=edge_index, device='cpu')
    
    dataloader = MockDataLoader()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    train_loss = trainer.train_epoch(dataloader, optimizer)
    assert train_loss > 0.0
