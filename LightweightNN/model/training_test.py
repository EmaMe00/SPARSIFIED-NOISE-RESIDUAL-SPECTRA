from package import *

# Funzione di addestramento
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start_time = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Calcola l'accuratezza
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        batch_end_time = time.time()
        batch_elapsed_time = batch_end_time - batch_start_time
        total_elapsed_time = batch_end_time - start_time
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t  Loss: {loss.item():.6f},  Batch Time: {batch_elapsed_time:.2f} seconds,  Total Time: {total_elapsed_time:.2f} seconds', end='\r')

    # Calcola l'accuratezza media per l'epoca
    accuracy = 100. * correct / total
    # Tempo totale per tutta l'epoca
    epoch_end_time = time.time()
    total_epoch_time = epoch_end_time - start_time

    print(f'\nTrain Epoch: {epoch} - Average loss: {epoch_loss / len(train_loader):.6f}, Accuracy: {accuracy:.2f}%, Total Time: {total_epoch_time:.2f} seconds')

    return epoch_loss / len(train_loader), accuracy  

# Funzione di test o validazione
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    start_time = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    end_time = time.time()  
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test/Validation set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%) - Time: {end_time - start_time:.2f} seconds')
    return test_loss, accuracy 