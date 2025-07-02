# predictor.py
import torch
from opt_train import BestUnitPredictor

def load_model(path):
    checkpoint = torch.load(path, map_location='cpu')  # safer for deployment
    model = BestUnitPredictor(78)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['unit_to_idx'], checkpoint['idx_to_unit']

def predict_best_units(model, unit_to_idx, idx_to_unit, input_units, top_k=3):
    try:
        input_ids = [unit_to_idx[u] for u in input_units]
        input_tensor = torch.tensor([input_ids])
        with torch.no_grad():
            logits = model(input_tensor)
            top_indices = torch.topk(logits, k=top_k, dim=1).indices[0].tolist()
            return [idx_to_unit[idx] for idx in top_indices]
    except KeyError as e:
        raise ValueError(f"Unknown unit name: {e}")


# app.py
from flask import Flask, request, render_template

app = Flask(__name__)
# Load model and unit lists
model, unit_to_idx, idx_to_unit = load_model("optimiser.pt")
all_units = sorted(unit_to_idx.keys())  # This is your list of units

@app.route("/", methods=["GET", "POST"])
def index():
    selected_units = {}

    if request.method == "POST":
        input_units = []
        for i in range(10):
            selected = request.form.get(f"unit{i}")
            if selected:
                input_units.append(selected)
            selected_units[f"unit{i}"] = selected  # save selection

        if len(input_units) < 1:
            return render_template("index.html", unit_list=all_units,
                                   selected_units=selected_units,
                                   error="Please select at least one unit.")

        try:
            prediction = predict_best_units(model, unit_to_idx, idx_to_unit, input_units)
            return render_template("index.html", unit_list=all_units,
                                   selected_units=selected_units,
                                   prediction=prediction)
        except ValueError as e:
            return render_template("index.html", unit_list=all_units,
                                   selected_units=selected_units,
                                   error=str(e))

    return render_template("index.html", unit_list=all_units, selected_units=selected_units)

if __name__ == "__main__":
    app.run(debug=True)
