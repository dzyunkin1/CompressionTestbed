import torch


def transform_dtype(dtype_str):
    if not isinstance(dtype_str, str):
        raise TypeError("Input must be a string")

    dtype_map = {
        "qint8": torch.qint8,
        "quint8": torch.quint8,
        "qint32": torch.qint32,
        # Add more dtype mappings as needed
    }

    if dtype_str in dtype_map:
        return dtype_map[dtype_str]
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
