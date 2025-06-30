## TFT Board Optimiser

This project builds a machine learning pipeline to recommend the best unit to add to a given board in **Teamfight Tactics (TFT)**. It uses real match data from top-tier players, retrieved via Riot API, and trains a PyTorch model to predict the optimal additional unit.

---

## Methodology

1. Data Collection

    Match data is retrieved from the Riot Games API for Challenger-tier players across multiple servers, specifically NA, KR, EUW, and SG. Data is pulled from multiple endpoints due to rate restrictions on individual endpoints, as well as for diversity.

    For each player, the most recent matches are pulled, and detailed match information is extracted and flattened into structured rows.

    Boards (unit compositions) and player placements are the key focus.

2. Data Preprocessing

    A custom scikit-learn pipeline is used to:

        Drop irrelevant, empty, or low-signal columns (e.g., companion info, player ID).

        Retain only final boards with first-place finishes for optimal training signals.

        Standardize unit names by stripping set prefixes (e.g., TFT14_ → Seraphine).

    The final dataset consists of rows representing endgame unit compositions.

3. Training Data Generation

    From each 1st-place board:

        One unit is removed. 

        The removed unit is used as the label, while the remaining units form the input, unless the removed unit is a summon which is non-purchasable in the shop. In the context of Set 14, the summon refers to the unit spawned by the Nitro trait.

        This mimics the recommendation task: “Given an incomplete board, which unit would best complete it?”

        For the training data, the removed unit is replaced and the process is repeated until each unit has been removed.

        In total, the number of training pairs generated per board is equal to the board size minus any summons.

5. Modeling

    A neural network model is trained using PyTorch.

    Inputs are sequences of unit names mapped to indices and passed through an embedding layer.

    The model learns to predict the missing unit via a classification head.

6. Evaluation

    The dataset is split into training and validation sets.

    During validation, only one pair is generated per board to reduce redundancy and prevent data leakage.

    Top-3 predicted units are output visually as there can be multiple viable options depending on the game state.

---

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/cieltheseal/tft1.git
   cd tft1

2. Install dependencies:
   pip install -r requirements.txt

3. Run ETL.py, opt_train.py, and opt_run.py in that order.
