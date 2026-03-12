import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "intfloat/multilingual-e5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed_query(query):

    text = "query: " + query

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**inputs)

        embeddings = average_pool(
            outputs.last_hidden_state,
            inputs["attention_mask"]
        )

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()