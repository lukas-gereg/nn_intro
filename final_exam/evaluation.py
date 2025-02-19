import torch


class Evaluation:
    @staticmethod
    def evaluate(loss, test_loader, model, device):
        with torch.no_grad():
            total_loss = 0
            results = []
            for test_loader_data, labels in test_loader:
                test_loader_data = test_loader_data.to(device)
                labels = labels.to(device)

                outputs = model(test_loader_data).squeeze(1)
                total_loss += loss(outputs, labels)

                results.extend(zip(labels, outputs))

            return total_loss / len(test_loader), results

