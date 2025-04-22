from torch.utils.data import DataLoader

from src.data.AudioScriptDataset import AudioScriptDataset
from src.data.collator import WhisperDataCollatorWhithPadding


def prepare_datasets(data_name_list, data_dir, tokenizer, batch_size):
    train_loader_list = []
    for name in data_name_list:
        train_dataset = AudioScriptDataset(
            data_dir=f"{data_dir}/{name}",
            tokenizer=tokenizer,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=WhisperDataCollatorWhithPadding(),
        )
        train_loader_list.append(train_loader)
    return train_loader_list


def prepare_test_dataset(data_dir, tokenizer, batch_size):
    test_native_dataset = AudioScriptDataset(
        data_dir=f"{data_dir}/test",
        tokenizer=tokenizer,
    )
    return DataLoader(
        test_native_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=WhisperDataCollatorWhithPadding(),
    )
