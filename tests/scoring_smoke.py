"""Offline numerical verification of standalone fp32 scoring, with random weights."""
import tempfile
import math
import torch
from tokenizers import Tokenizer, decoders, pre_tokenizers, models, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from ladder.scoring import Scorer, Request, score_pairs


def main():
    torch.manual_seed(4)  # Fixture initialization only; evaluation never samples.
    torch.set_num_threads(2)
    text = 'Oni piszą list. Ona pisze list. Żółć i źdźbło. The bird sings.'
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator([text], trainers.BpeTrainer(vocab_size=300, special_tokens=['<eos>'],
                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>', bos_token='<eos>')
    with tempfile.TemporaryDirectory() as temp:
        tokenizer.save_pretrained(temp)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_positions=64, n_embd=32,
                                         n_layer=1, n_head=2, bos_token_id=0, eos_token_id=0))
        model.save_pretrained(temp)
        scorer = Scorer(temp, batch_size=4, context=64)
        request = Request('Żółć i źdźbło.')
        first, second = scorer.score([request]), scorer.score([request])
        assert first == second, 'Repeated evaluations must match exactly'
        math_scorer = Scorer(temp, batch_size=4, context=64, attention='sdpa_math')
        math_result = math_scorer.score([request])
        assert math_scorer.metadata['attention'] == 'sdpa_math'
        assert abs(math_result[0]['nll_nats'] - first[0]['nll_nats']) < 1e-4
        del math_scorer
        assert first[0]['bytes'] == len(request.text.encode())
        ids = [tokenizer.bos_token_id] + tokenizer.encode(request.text, add_special_tokens=False)
        with torch.inference_mode():
            logits = scorer.model(torch.tensor([ids[:-1]]), use_cache=False).logits
            expected = torch.nn.functional.cross_entropy(logits[0], torch.tensor(ids[1:]), reduction='sum').item()
        assert abs(first[0]['nll_nats'] - expected) < 1e-4, (first, expected)
        long = scorer.score([Request(text * 8)])[0]
        assert long['tokens'] == len(tokenizer.encode(text * 8, add_special_tokens=False))
        assert long['bytes'] == len((text * 8).encode())
        pair = {'id': 'fixture', 'paradigm': 'agreement', 'language': 'pl',
                'good': 'Oni piszą list.', 'bad': 'Oni pisze list.',
                'region': {'good': [3, 9], 'bad': [3, 9]}}
        pairs = score_pairs(scorer, [pair])
        assert 0 <= pairs[0]['sentence_prob'] <= 1
        assert 0 <= pairs[0]['region_prob'] <= 1
        assert all(p.dtype == torch.float32 for p in scorer.model.parameters())
        shapes = []
        hook = scorer.model.register_forward_pre_hook(lambda module, args, kwargs: shapes.append(tuple(kwargs['input_ids'].shape)), with_kwargs=True)
        scorer.score([Request('Oni piszą list.')] * 5)
        hook.remove()
        assert shapes and set(shapes) == {(4, 64)}, shapes
        # Exercise a real frozen-corpus report; partial coverage cannot enable decisions.
        import json
        from pathlib import Path
        from ladder.corpus import freeze, SLICES
        from ladder.benchmark import evaluate_benchmark
        root = Path(temp)
        config = {'slices': {s: {'sources': [s], 'out_of_mix': i < 2} for i, s in enumerate(SLICES)},
                  'micro_slices': list(SLICES[:2])}
        (root / 'sources.json').write_text(json.dumps(config))
        docs = [{'id': s, 'slice': s, 'source': s,
                 'text': ' '.join(s + str(i) for i in range(30))} for s in SLICES]
        (root / 'candidates.jsonl').write_text(''.join(json.dumps(d) + '\n' for d in docs))
        (root / 'train.jsonl').write_text(json.dumps({'id': 'train', 'source': 'train', 'text': 'other material'}))
        freeze(root / 'sources.json', [root / 'train.jsonl'], root / 'candidates.jsonl', root / 'corpus',
               tokenizer, {'fixture': True}, tokens_per_slice=20, micro_per_slice=10)
        report = evaluate_benchmark(scorer, root / 'corpus', tier='micro', rung=8, allow_incomplete=True)
        assert not report['decision_eligible'] and report['missing']
        assert 'bpb/out_of_mix' in report['metrics']
        assert len(report['corpus']) == 3  # Two micro slices plus the pooled out-of-mix score.
        mc = [{'id': str(i), 'task': 'task' + str(i), 'prefix': 'Oni', 'correct': ' piszą'} for i in range(3)]
        (root / 'mc.jsonl').write_text(''.join(json.dumps(d) + '\n' for d in mc))
        report = evaluate_benchmark(scorer, root / 'corpus', tier='fast', rung=50,
                                    mc_path=root / 'mc.jsonl', allow_incomplete=True)
        assert not report['decision_eligible']
        assert len(report['mc_items']) == 3
        assert all('mc_bpb/task' + str(i) in report['metrics'] for i in range(3))
        print('Standalone scorer passed: manual NLL reference, repeated exact equality, Unicode bytes, long-window coverage, sentence/region probabilities, fixed (4,64) shapes, fp32.')


if __name__ == '__main__':
    main()
