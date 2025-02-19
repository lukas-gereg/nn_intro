import copy as cp

from validation import Validation


class Training:
    @staticmethod
    def train(epochs, device, optimizer, model, loss, train_loader, validation_loader, threshold):
        old_validation_value = Validation.validate(validation_loader, device, model, loss)
        counter = 0
        best_weights = cp.deepcopy(model.state_dict())
        losses = []

        for i in range(epochs):
            epoch_loss = 0

            for batch_idx, (data, labels) in enumerate(train_loader):
                data = data.to(device)
                labels = labels.to(device)

                prediction = model(data)
                print("pred: ", prediction)
                prediction = prediction
                print(prediction, labels)
                
                current_loss = loss(prediction, labels)

                optimizer.zero_grad()
                current_loss.backward()
                optimizer.step()
                epoch_loss += current_loss.item()

            print(f'epoch {i + 1}, loss per item: {epoch_loss / len(train_loader)}')

            current_validation_value = Validation.validate(validation_loader, device, model, loss)

            losses.append(current_validation_value)

            if current_validation_value <= old_validation_value:
                old_validation_value = current_validation_value
                counter = 0
                best_weights = cp.deepcopy(model.state_dict())
            else:
                if counter < threshold:
                    counter += 1
                else:
                    model.load_state_dict(best_weights)
                    print(f"Risk of over fitting parameters, ending learning curve at iteration {i + 1}")
                    return losses[: -counter]

        model.load_state_dict(best_weights)
        return losses
