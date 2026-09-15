"""
Shared model loader for pressure-testing experiment.

Models:
    mistral   -> Mistral-Small-3.2-24B-Instruct-2506
    magistral -> Magistral-Small-2509

Both models use the same 4-bit NF4 BitsAndBytes configuration.
"""

import os

import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Mistral3ForConditionalGeneration,
)

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
load_dotenv()


MODEL_IDS = {
    "mistral": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "magistral": "mistralai/Magistral-Small-2509",
}


DOWNLOAD_IGNORE_PATTERNS = [
    "consolidated.safetensors",
    "*.pth",
    "*.bin",
]


BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


_loaded_cache = {}


def ensure_downloaded(model_key: str) -> str:
    """Ensure the model exists in the local Hugging Face cache."""

    if model_key not in MODEL_IDS:
        raise ValueError(
            f"Unknown model_key '{model_key}'. "
            f"Expected one of: {list(MODEL_IDS.keys())}"
        )

    repo_id = MODEL_IDS[model_key]

    local_path = snapshot_download(
        repo_id=repo_id,
        ignore_patterns=DOWNLOAD_IGNORE_PATTERNS,
        token=os.environ.get("HF_TOKEN"),
    )

    return local_path


def load_mistral_tokenizer(local_path: str):
    """
    Load the official Mistral tokenizer.
    """

    print("Loading official Mistral tokenizer...")

    tokenizer = MistralTokenizer.from_file(
        os.path.join(local_path, "tekken.json")
    )

    return tokenizer


def load_model(model_key: str):
    """
    Load model and tokenizer.

    Returns:
        model, tokenizer
    """

    if model_key not in MODEL_IDS:
        raise ValueError(
            f"Unknown model_key '{model_key}'. "
            f"Expected one of: {list(MODEL_IDS.keys())}"
        )

    if model_key in _loaded_cache:
        return _loaded_cache[model_key]

    repo_id = MODEL_IDS[model_key]

    print()
    print("=" * 80)
    print(f"Loading {repo_id}")
    print("=" * 80)

    print("Checking Hugging Face cache...")
    local_path = ensure_downloaded(model_key)

    print(f"Model path: {local_path}")

    # ------------------------------------------------------------------
    # MISTRAL
    # ------------------------------------------------------------------

    if model_key == "mistral":

        tokenizer = load_mistral_tokenizer(local_path)

        print("Loading Mistral model in 4-bit NF4...")

        model = Mistral3ForConditionalGeneration.from_pretrained(
            local_path,
            quantization_config=BNB_CONFIG,
            device_map="cuda",
            tie_word_embeddings=False,
        )

    # ------------------------------------------------------------------
    # MAGISTRAL
    # ------------------------------------------------------------------

    elif model_key == "magistral":

        print("Loading official Mistral tokenizer for Magistral...")

        tokenizer = load_mistral_tokenizer(local_path)

        print("Loading Magistral model in 4-bit NF4...")

        model = Mistral3ForConditionalGeneration.from_pretrained(
            local_path,
            quantization_config=BNB_CONFIG,
            device_map="cuda",
            tie_word_embeddings=False,
        )

    else:
        raise ValueError(f"Unsupported model_key: {model_key}")

    model.eval()

    _loaded_cache[model_key] = (model, tokenizer)

    print("Model loaded successfully.")

    return model, tokenizer


def build_mistral_prompt(tokenizer, messages):
    """
    Convert our normal messages format into Mistral's official
    ChatCompletionRequest format.

    Supports:
        - system
        - user
        - assistant

    This preserves the full conversation history required by
    the Phase 2 pressure-testing design.
    """

    mistral_messages = []

    for message in messages:

        role = message["role"]
        content = message["content"]

        if role == "system":

            mistral_messages.append(
                SystemMessage(
                    content=content
                )
            )

        elif role == "user":

            mistral_messages.append(
                UserMessage(
                    content=content
                )
            )

        elif role == "assistant":

            mistral_messages.append(
                AssistantMessage(
                    content=content
                )
            )

        else:

            raise ValueError(
                f"Unsupported message role: {role}"
            )

    request = ChatCompletionRequest(
        messages=mistral_messages,
    )

    encoded = tokenizer.encode_chat_completion(
        request
    )

    return encoded.tokens


def generate(
    model_key: str,
    messages: list[dict],
    max_new_tokens: int = 400,
    temperature: float = 0.0,
) -> tuple[str, str | None]:
    """
    Generate a response.

    Returns:
        final_text, reasoning_text_or_none
    """

    model, tokenizer = load_model(model_key)

    # ------------------------------------------------------------------
    # MISTRAL
    # ------------------------------------------------------------------

    if model_key == "mistral":

        input_tokens = build_mistral_prompt(
            tokenizer,
            messages,
        )

        input_ids = torch.tensor(
            [input_tokens],
            dtype=torch.long,
            device=model.device,
        )

        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():

            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature
                if temperature > 0.0
                else None,
            )

        generated_ids = output_ids[
            0,
            input_ids.shape[1]:
        ]

        full_text = tokenizer.decode(
            generated_ids.tolist()
        ).strip()

    # ------------------------------------------------------------------
    # MAGISTRAL
    # ------------------------------------------------------------------

    elif model_key == "magistral":

        input_tokens = build_mistral_prompt(
            tokenizer,
            messages,
        )

        input_ids = torch.tensor(
            [input_tokens],
            dtype=torch.long,
            device=model.device,
        )

        attention_mask = torch.ones_like(input_ids)

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
        }

        if temperature > 0.0:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():

            output_ids = model.generate(
                **generation_kwargs
            )

        generated_ids = output_ids[
            0,
            input_ids.shape[1]:
        ]

        full_text = tokenizer.decode(
            generated_ids.tolist()
        ).strip()

    else:
        raise ValueError(
            f"Unsupported model_key: {model_key}"
        )

    # ------------------------------------------------------------------
    # Extract reasoning if present
    # ------------------------------------------------------------------

    reasoning = None

    if "<think>" in full_text and "</think>" in full_text:

        start = (
            full_text.index("<think>")
            + len("<think>")
        )

        end = full_text.index("</think>")

        reasoning = full_text[start:end].strip()

        full_text = full_text[
            end + len("</think>"):
        ].strip()

    return full_text, reasoning


def test_model(model_key: str):

    print()
    print("=" * 80)
    print(f"TESTING: {model_key}")
    print("=" * 80)

    messages = [
        {
            "role": "system",
            "content": (
                "Respond with valid JSON only: "
                '{"answer": "<value>"}'
            ),
        },
        {
            "role": "user",
            "content": (
                "What is 2+2? "
                "Respond with the JSON object only."
            ),
        },
    ]

    text, reasoning = generate(
        model_key=model_key,
        messages=messages,
        max_new_tokens=100,
        temperature=0.0,
    )

    print()
    print("OUTPUT:")
    print(text)

    if reasoning:
        print()
        print(
            f"REASONING CAPTURED: "
            f"{len(reasoning)} characters"
        )


if __name__ == "__main__":

    import sys

    if len(sys.argv) > 1:

        key = sys.argv[1]

        if key not in MODEL_IDS:
            raise SystemExit(
                f"Unknown model '{key}'. "
                f"Use 'mistral' or 'magistral'."
            )

        test_model(key)

    else:

        print(
            "Usage:\n"
            "  python model_loader.py mistral\n"
            "  python model_loader.py magistral"
        )
