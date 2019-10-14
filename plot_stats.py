import torch
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="metrics plotting script")
parser.add_argument('--stat', type=str, required=True,
                    help='the path to the stats file(.st)')

if __name__ == "__main__":
    options = parser.parse_args()
    results = torch.load(options.stat)

    metrics = {k: [] for k in results[0].keys()}
    for m in results.values():
        for k, v in m.items():
            metrics[k].append(v.avg)

    metrics['classifier_loss'] = metrics.pop('loss_classifier')

    for idx, (k, values) in enumerate(metrics.items()):
        plt.figure(num=idx + 1)
        plt.plot(list(range(len(values))), values, 'b')
        plt.title(k)
        plt.xlabel('epochs')
        plt.title(f"{k} over epochs")
        plt.ylabel(k)
    plt.show()
