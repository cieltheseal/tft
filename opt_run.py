import torch
from opt_train import BestUnitPredictor

def load_model(path, model_class, *args, **kwargs):
    checkpoint = torch.load(path)
    model = model_class(*args, **kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['unit_to_idx'], checkpoint['idx_to_unit']

checkpoint = torch.load('optimiser.pt')

#-------------------------------
# Run model
#-------------------------------

# Load model
model = BestUnitPredictor(78)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

unit_to_idx = checkpoint['unit_to_idx']
idx_to_unit = checkpoint['idx_to_unit']

# Test prediction
model.eval()
input_units = ["Ekko", "Zyra", "DrMundo", "Neeko", "Rengar", "Ziggs"]
input_ids = [unit_to_idx[u] for u in input_units]
input_tensor = torch.tensor([input_ids])

with torch.no_grad():
    logits = model(input_tensor)
    top3_indices = torch.topk(logits, k=3, dim=1).indices[0].tolist()
    top3_units = [idx_to_unit[idx] for idx in top3_indices]
    print("Top 3 predicted best units:", top3_units)