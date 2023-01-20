from pathlib import Path
from argparse import Namespace
from typing import Union, List
from fastprogress import progress_bar
from typing_extensions import TypedDict
import torch
from enum import Enum
from datasets import Dataset, DatasetDict, load_metric
import numpy as np
from transformers import DataCollatorForTokenClassification, AutoTokenizer, AutoModelForTokenClassification, \
    TrainingArguments, Trainer, RobertaTokenizerFast

percent = "percent"
partitive = "partitive"
Current_task = percent

if (Current_task == percent):
    train_data_path = "data/%_nombank.clean.train"
    valid_data_path = "data/%_nombank.clean.dev"
    test_data_path = "data/%_nombank.clean.test"
elif(Current_task == partitive):
    train_data_path = "data/partitive_group_nombank.clean.train"
    valid_data_path = "data/partitive_group_nombank.clean.dev"
    test_data_path = "data/partitive_group_nombank.clean.test"
else:
    print("ERROR:please choose a dataset.")
    exit(1)

SYMBOL_DICT = {
    "COMMA": ",",
}

LABEL_LIST = ["NONE", "PRED", "ARG1", "SUPPORT"]
POS_LIST = ["CC", "CD", "DT", "FW", "IN", "JJ", "JJR", "JJS",
            "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT",
            "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "SYM",
            "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP",
            "VBZ", "WDT", "WP", "WP$", "WRB", "PU", "EX",
            "RP"]

CONVERSION= {
    ".": "PU",
    ",": "PU",
    "COMMA": "PU",
    "$": "PU",
    ":": "PU",
    "(": "PU",
    ")": "PU",
    "``": "PU",
    "''": "PU",
    "#": "PU"
}
BIO_TAG_LIST = ["O", "B-NP", "I-NP", "B-VP", "I-VP", "B-PP",
                "I-PP", "B-ADJP", "I-ADJP", "B-ADVP", "I-ADVP",
                "B-SBAR", "I-SBAR", "B-PRT", "I-PRT", "B-CONJP",
                "I-CONJP", "B-UCP", "I-UCP"]


class Word(TypedDict):
    word: str
    pos: str
    biotag: str
    label: Union[str, None]

def parse_input2(input_file, drop_label=False):
    data_dict = {'tokens': [], 'partitive_roles': [], 'pos_tags': [], 'bio_tags': []}
    with open(input_file, "r") as f:
        lines = f.readlines()
    this_token=[]
    this_role=[]
    this_pos=[]
    this_bio=[]
    for line in lines:
        line = line.strip()
        word_info = line.split("\t")
        if len(word_info) >= 5:
            word = word_info[0].strip()
            if word == "COMMA":
                word = ","

            pos = word_info[1].strip()
            if pos in CONVERSION:
                pos = CONVERSION[pos]
            if pos not in POS_LIST:
                pos = "PU"
            pos=POS_LIST.index(pos)

            bio = word_info[2].strip()
            if bio not in BIO_TAG_LIST:
                bio = "O"
            bio=BIO_TAG_LIST.index(bio)

            if len(word_info) >= 6:
                label = word_info[5].strip()
            else:
                label = "NONE"

            if label not in LABEL_LIST:
                label = "NONE"
            label=LABEL_LIST.index(label)

            this_token.append(word)
            this_role.append(label)
            this_pos.append(pos)
            this_bio.append(bio)
        else:
            data_dict['tokens'].append(this_token)
            data_dict['partitive_roles'].append(this_role)
            data_dict['pos_tags'].append(this_pos)
            data_dict['bio_tags'].append(this_bio)
            this_token = []
            this_role = []
            this_pos = []
            this_bio = []
    return Dataset.from_dict(data_dict)

def parse_input(input_file: Union[str, Path], drop_label=False) -> List[Union[List[Word], None]]:
    """
    Parses the input file and returns a list of lists of words.
    """
    with open(input_file, "r") as f:
        lines = f.readlines()
    sentences: List[Union[List[Word], None]] = []
    last_sentence: List[Word] = []
    print("Parsing input file lines...")
    line_no = 0
    for line in progress_bar(lines):
        line_no += 1
        line = line.strip()
        word_info = line.split("\t")
        if len(word_info) >= 5:
            word_str = word_info[0].strip()
            if word_str in SYMBOL_DICT:
                word_str = SYMBOL_DICT[word_str]
            pos = word_info[1].strip()
            if pos in CONVERSION:
                pos = CONVERSION[pos]
            if pos not in POS_LIST:
                print(f"Warning: invalid POS on line {line_no} \"{pos}\", treated as PU.")
                pos = "PU"
            biotag = word_info[2].strip()
            if biotag not in BIO_TAG_LIST:
                print(f"Warning: invalid bio tag on line {line_no} \"{biotag}\", treated as O.")
                biotag = "O"
            if len(word_info) >= 6:
                label = word_info[5].strip()
            else:
                label = "NONE"
            if label not in LABEL_LIST:
                print(f"Warning: invalid label on line {line_no} \"{label}\", treated as NONE.")
                label = "NONE"
            if drop_label:
                label = None
            word = Word(word=word_str, pos=pos, biotag=biotag, label=label)
            last_sentence.append(word)
        else:
            if len(last_sentence) > 0:
                sentences.append(last_sentence)
            last_sentence = []
    if len(last_sentence) > 0:
        sentences.append(last_sentence)
    return sentences


print("Importing pretrained model...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer=RobertaTokenizerFast(add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(LABEL_LIST))


def build_dataset_from_sentences(sentences, drop_label=False):
    dataset_tokens = []
    dataset_partitive_roles = []
    dataset_pos_tags = []
    dataset_bio_tags = []
    for sentence in sentences:
        tokens = [word['word'] for word in sentence]
        if not drop_label:
            partitive_roles = [LABEL_LIST.index(word['label']) for word in sentence]
        else:
            partitive_roles = None
        pos_tags = [POS_LIST.index(word['pos']) for word in sentence]
        bio_tags = [BIO_TAG_LIST.index(word['biotag']) for word in sentence]
        dataset_tokens.append(tokens)
        if not drop_label:
            dataset_partitive_roles.append(partitive_roles)
        dataset_pos_tags.append(pos_tags)
        dataset_bio_tags.append(bio_tags)
    if not drop_label:
        dataset_dict = {
            "tokens": dataset_tokens,
            "partitive_roles": dataset_partitive_roles,
            "pos_tags": dataset_pos_tags,
            "bio_tags": dataset_bio_tags
        }

    else:
        dataset_dict = {
            "tokens": dataset_tokens,
            "pos_tags": dataset_pos_tags,
            "bio_tags": dataset_bio_tags
        }
    return Dataset.from_dict(dataset_dict)

print("Building datasets...")
train_raw_dataset = parse_input2(train_data_path)
valid_raw_dataset = parse_input2(valid_data_path)
test_raw_dataset = parse_input2(test_data_path, drop_label=True)

print("Parsing input...")
# train_sentences = parse_input(train_data_path)
# valid_sentences = parse_input(valid_data_path)
# test_sentences = parse_input(test_data_path, drop_label=True)
# train_raw_dataset = build_dataset_from_sentences(train_sentences)
# valid_raw_dataset = build_dataset_from_sentences(valid_sentences)
# test_raw_dataset = build_dataset_from_sentences(test_sentences, drop_label=True)

raw_datasets = DatasetDict(train=train_raw_dataset, valid=valid_raw_dataset, test=test_raw_dataset)

label_all_tokens = True

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True)

    if "partitive_roles" in examples:
        labels = []
        for i, label in enumerate(examples["partitive_roles"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("Tokenizing datasets...")
tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


batch_size = 64

print("Training...")
train_args = TrainingArguments(
    "bert_output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01
)

data_collator = DataCollatorForTokenClassification(tokenizer)

metric = load_metric("seqeval")

trainer = Trainer(
    model,
    train_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()

print("Predicting...")
test_results = trainer.predict(tokenized_datasets['test'])

out = []

for i in range(len(tokenized_datasets['test'])):
    sentence = tokenized_datasets['test'][i]
    tokenized_input = tokenizer(sentence["tokens"], truncation=True, is_split_into_words=True)
    predictions = test_results.predictions[i]
    label_ids = []
    max_arg1_index = -1
    max_arg1_value = -100
    for j, prediction in enumerate(predictions):
        arg1_value = prediction[2]
        if arg1_value > max_arg1_value:
            max_arg1_value = arg1_value
            max_arg1_index = j
    for j, prediction in enumerate(predictions):
        label_ids.append(2 if j == max_arg1_index else 0)
    word_id_to_label_idx = {}
    for j, word_id in enumerate(tokenized_input.word_ids()):
        if word_id in word_id_to_label_idx or word_id is None:
            continue
        word_id_to_label_idx[word_id] = j
    labelings = []
    has_arg1 = False

    for j, token in enumerate(sentence["tokens"]):
        label_idx = word_id_to_label_idx[j]
        next_label_idx = word_id_to_label_idx[j + 1] if j + 1 < len(word_id_to_label_idx) else len(label_ids)

        is_arg1 = False

        for k in range(label_idx, next_label_idx):
            label_id = label_ids[k]
            if label_id == 2:
                is_arg1 = True

        label = 'ARG1' if is_arg1 else None
        if label == 'ARG1':
            has_arg1 = True
        if token == ',':
            token = 'COMMA'
        labelings.append((token, label))
    out.append(labelings)

    if not has_arg1:
        print(f"line {i} has no arg1")

arg1_count = 0
for labelings in out:
    has_arg1 = 0
    for token, label in labelings:
        if label == 'ARG1':
            has_arg1 += 1
    if has_arg1 == 1:
        arg1_count += 1
print(arg1_count)

arg1_count = 0

out_path = "result/"+Current_task +".txt"
with open(out_path, 'w') as f:
    for line in out:
        for labling in line:
            if labling[1] and labling[1] == 'ARG1':
                f.write(f"{labling[0]}\t{labling[1]}\n")
                arg1_count += 1
            else:
                f.write(f"{labling[0]}\n")
        f.write("\n")
