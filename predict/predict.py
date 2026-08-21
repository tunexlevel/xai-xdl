import torch
import json
from model import Seq2SeqGRU
from utils import smiles_to_indices
from rdkit import Chem
from rdkit.Chem import Draw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load vocab and model ===
with open("token2idx.json") as f:
    token2idx = json.load(f)
with open("idx2token.json") as f:
    idx2token = {int(k): v for k, v in json.load(f).items()}

print("Loaded vocab with tokens:", list(token2idx.keys()))
print("idx2token mapping:", idx2token)

pad_idx = token2idx["<pad>"]
sos_idx = token2idx["<sos>"]
eos_idx = token2idx["<eos>"]
unk_idx = token2idx["<unk>"]

model = Seq2SeqGRU(
    input_dim=len(token2idx),
    output_dim=len(token2idx),
    emb_dim=128,
    hidden_dim=256,
    pad_idx=pad_idx
)
model.load_state_dict(torch.load("reaction_model.pt", map_location=device))
model.to(device)
model.eval()

# === Prediction ===
def predict_product(reactant_smiles, max_len=120):
    input_ids = smiles_to_indices(reactant_smiles, token2idx, max_len)
    input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        embedded_src = model.embedding(input_tensor)
        _, hidden = model.encoder(embedded_src)

        input_token = torch.tensor([[sos_idx]], device=device)
        pred_tokens = []
        pred_ids_list = []

        for _ in range(max_len):
            embedded_tgt = model.embedding(input_token)
            output, hidden = model.decoder(embedded_tgt, hidden)
            logits = model.fc_out(output)
            pred_id = logits.argmax(2).item()
            pred_ids_list.append(pred_id)

            if pred_id == eos_idx:
                break

            pred_tokens.append(idx2token[pred_id])
            input_token = torch.tensor([[pred_id]], device=device)

    print("DEBUG: pred_ids =", pred_ids_list)
    print("DEBUG: pred_tokens =", pred_tokens)

    pred_smiles = "".join(pred_tokens).replace("<pad>", "")
    return pred_smiles

# === Example ===
if __name__ == "__main__":
    from utils import tokenize_smiles
    reactant = "CC(=O)OC1=CC=CC=C1C(=O)O"
    predicted = predict_product(reactant)
    print(f"Reactant:  {reactant}")
    print(f"Predicted: {predicted}")

    # try:
    #     mol1 = Chem.MolFromSmiles(reactant)
    #     mol2 = Chem.MolFromSmiles(predicted)
    #     Draw.MolsToGridImage([mol1, mol2], legends=["Reactant", "Predicted Product"]).show()
    # except Exception as e:
    #     print("Visualization failed:", e)
