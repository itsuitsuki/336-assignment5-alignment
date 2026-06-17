from datasets import load_dataset, concatenate_datasets

# Method 1: Load all configurations explicitly
configs = ["harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"]
datasets = {}

# for config in configs:
#     datasets[config] = load_dataset("Anthropic/hh-rlhf", data_dir=config)
#     print(f"\n=== {config.upper()} ===")
#     print(datasets[config])

# # Method 2: Load each config and print examples
# for config in configs:
#     dataset = load_dataset("Anthropic/hh-rlhf", data_dir=config)
#     print(f"\n=== {config.upper()} ===")
#     for split_name, split_data in dataset.items():
#         print(f"{split_name}: {len(split_data)} examples")
#         if len(split_data) > 0:
#             print(f"Example: {split_data[0]}")

subsets = ["harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"]
hh_dataset = concatenate_datasets(
    [load_dataset("Anthropic/hh-rlhf", data_dir=subset)["train"] for subset in subsets]
)
print(hh_dataset)