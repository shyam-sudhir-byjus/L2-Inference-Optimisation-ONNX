# # Load pretrained model and tokenizer
from transformers import AutoModel, AutoTokenizer
from torch.nn import functional as F

model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

model = AutoModel.from_pretrained(model_name)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    temp = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return F.normalize(temp, p=2, dim=1)


# Get the first example data to run the model and export it to ONNX

sample = ["Hey, how are you today?"]
inputs = tokenizer(sample, padding=True, truncation=True, return_tensors="pt")

## Convert Model to ONNX Format
import os
import torch

device = torch.device("cpu")

# Set model to inference mode, which is required before exporting
# the model because some operators behave differently in
# inference and training mode.
model.eval()
model.to(device)

output_dir = os.path.join(".", "onnx_models")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

export_model_path = os.path.join(output_dir, "all_MiniLM_L6-v2.onnx")


with torch.no_grad():
    symbolic_names = {0: "batch_size", 1: "max_seq_len"}
    torch.onnx.export(
        model,  # model being run
        args=tuple(inputs.values()),  # model input (or a tuple for multiple inputs)
        f=export_model_path,  # where to save the model (can be a file or file-like object)
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=[
            "input_ids",  # the model's input names
            "attention_mask",
            "token_type_ids",
        ],
        output_names=["start", "end"],  # the model's output names
        dynamic_axes={
            "input_ids": symbolic_names,  # variable length axes
            "attention_mask": symbolic_names,
            "token_type_ids": symbolic_names,
            "start": symbolic_names,
            "end": symbolic_names,
        },
    )
    print("Model exported at ", export_model_path)
