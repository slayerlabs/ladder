"""Offline integration smoke: actual forward passes, random weights, no capability claim."""
import tempfile
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from ladder.tasks import suite
from ladder.evaluation import evaluate


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    items = suite(4, levels=(1,))
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator([i.prompt + ' ' + ' '.join(i.choices) for i in items],
                                  WordLevelTrainer(special_tokens=["[UNK]", "[EOS]", "[PAD]"]))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]")
    with tempfile.TemporaryDirectory() as temp:
        fast.save_pretrained(temp)
        config = GPT2Config(vocab_size=len(fast), n_positions=256, n_embd=32, n_layer=1, n_head=2,
                            bos_token_id=fast.eos_token_id, eos_token_id=fast.eos_token_id,
                            pad_token_id=fast.pad_token_id)
        GPT2LMHeadModel(config).save_pretrained(temp)
        report = evaluate(items, backend="hf", model_path=temp)
        assert len(report["items"]) == 28
        assert report["model"]["actual_parameters"] > 0
        assert len(report["model"]["tokenizer_sha256"]) == 64
        assert not report["model"]["synthetic_control"]
        # Verify oversized prompts are refused instead of silently truncated.
        from dataclasses import replace
        long = replace(items[0], prompt="apple " * 300)
        try:
            evaluate([long], backend="hf", model_path=temp)
        except ValueError as exc:
            assert "Context exceeds" in str(exc)
        else:
            raise AssertionError("Overflow must fail")
        print("HF integration passed: 28 items, 112 actual likelihoods; overflow rejected.")


if __name__ == "__main__":
    main()
