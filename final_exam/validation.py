import torch


class Validation:
    @staticmethod
    def validate(validation_loader, device, model, loss):
        with torch.no_grad():
            current_validation_value = 0

            for validation_loader_data, validation_labels in validation_loader:
                validation_loader_data = validation_loader_data.to(device)
                validation_labels = validation_labels.to(device)

                val_prediction = model(validation_loader_data).squeeze(1)

                validation_loss = loss(val_prediction, validation_labels)
                current_validation_value += validation_loss.item()

            return current_validation_value / len(validation_loader)
