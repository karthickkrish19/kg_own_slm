"""Quick test script to check model output quality."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from inference import SLMInference
from config import Config
import yaml

with open('config.yaml') as f:
    ov = yaml.safe_load(f)
cfg = Config(**{k: v for k, v in ov.items() if hasattr(Config, k)})
inf = SLMInference(cfg, 'out/checkpoints/best_model.pt', device='cpu')

prompts = [
    'The study of geometry',
    'A triangle has three',
    'Light travels through',
    'Water is heated in a boiler',
    'Newton discovered that',
]

print("\n" + "="*60)
print("  MODEL OUTPUT TEST")
print("="*60)

for p in prompts:
    out = inf.generate(p, max_new_tokens=100, temperature=0.7, top_k=40, top_p=0.92)
    print(f'\nPROMPT: {p}')
    print(f'OUTPUT: {out}')
    print('-'*60)

# Perplexity test
print("\n" + "="*60)
print("  PERPLEXITY TEST (lower = better)")
print("="*60)
test_texts = [
    'The refraction of light through a prism produces different colours.',
    'A triangle has three sides and three angles.',
    'The boiler generates steam by heating water under pressure.',
    'Random pizza laptop bluetooth astronaut quantum zebra unicorn.',
]
for t in test_texts:
    ppl = inf.perplexity(t)
    print(f'  ppl={ppl:7.1f}  |  {t}')
