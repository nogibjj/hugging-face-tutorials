from datasets import load_dataset
from pprint import pprint

nba_dataset = load_dataset("noahgift/social-power-nba")

print(f"Features: {nba_dataset['train'].features}\n")
# print out the attributes of one example
harden = nba_dataset["train"][1]
# pprint out all features of harden
pprint(harden)
