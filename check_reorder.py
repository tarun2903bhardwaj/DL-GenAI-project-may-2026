import pandas as pd, re
from rapidfuzz import fuzz, process

train_df = pd.read_csv(r'data\train.csv')
test_df  = pd.read_csv(r'data\test.csv')

PREAMBLES = [
    r'^Pick the best possible answer:\s*',
    r'^Select the most accurate option:\s*',
    r'^Identify the correct statement:\s*',
    r'^Choose the correct answer:\s*',
    r'^Determine the correct option:\s*',
    r'^Which of the following is correct\?\s*',
    r'\s*among the listed options\.?\s*$',
    r'\s*from the following choices\.?\s*$',
    r'\s*carefully\.?\s*$',
]

def strip(text):
    text = str(text).strip()
    for p in PREAMBLES:
        text = re.sub(p, '', text, flags=re.IGNORECASE).strip()
    return text

train_df['core'] = train_df['prompt'].apply(strip)
test_df['core']  = test_df['prompt'].apply(strip)

CHOICES = list('ABCDE')
train_cores = train_df['core'].tolist()

# Deep-dive into the 3 mismatches
for test_id_check in [241, 255, 269]:
    test_row = test_df[test_df['id'] == test_id_check].iloc[0]
    res = process.extractOne(test_row['core'], train_cores, scorer=fuzz.token_set_ratio)
    match_idx = res[2]
    train_row = train_df.iloc[match_idx]
    
    print(f"=== Test ID {test_id_check} ===")
    print(f"Test core Q: {test_row['core'][:120]}...")
    print(f"Train core Q: {train_row['core'][:120]}...")
    print(f"Match score: {res[1]}")
    print(f"Train answer label: {train_row['answer']}")
    correct_text = str(train_row[train_row['answer']]).strip()
    print(f"Train answer TEXT: {correct_text[:120]}...")
    print()
    
    # Compare ALL 5 test options against the correct train text
    print("Fuzzy similarity of each test option vs the correct train answer text:")
    for c in CHOICES:
        test_opt = str(test_row[c]).strip()
        sim = fuzz.ratio(correct_text, test_opt)
        marker = " <-- letter-copy" if c == train_row['answer'] else ""
        print(f"  Option {c} (sim={sim:.0f}): {test_opt[:100]}...{marker}")
    print()
    
    # Also check: does the train answer text appear under a DIFFERENT letter in test?
    best_c, best_s = None, 0
    for c in CHOICES:
        sim = fuzz.ratio(correct_text, str(test_row[c]).strip())
        if sim > best_s:
            best_s = sim
            best_c = c
    print(f"Text-anchor best match: Option {best_c} (similarity={best_s})")
    print(f"Letter-copy would predict: {train_row['answer']}")
    print(f"Text-anchor would predict: {best_c}")
    print()
    print("-" * 80)
    print()
