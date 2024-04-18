import torch


def peak_pick_max_mean(data: torch.tensor, mean_range: int = 7, max_range: int = 3):
    mean_filter = torch.nn.AvgPool1d(kernel_size=mean_range+1, stride=1, padding=0)
    max_filter = torch.nn.MaxPool1d(kernel_size=max_range+1, stride=1, padding=0)
    padded = torch.nn.functional.pad(data, (mean_range, 0))
    mean = mean_filter(padded)
    padded = torch.nn.functional.pad(mean, (max_range, 0))
    maximum = max_filter(padded)
    assert maximum.shape == mean.shape and maximum.shape == data.shape

    time = torch.tensor(range(data.shape[-1]))
    data[data < mean] = 0.0
    data[data < maximum] = 0.0
    out = []
    for i in range(data.shape[0]):
        out.append([])
        for j in range(data.shape[1]):
            out[i].append(torch.stack((time, data[i, j]))[:, data[i, j] > 0.0])
    return out
