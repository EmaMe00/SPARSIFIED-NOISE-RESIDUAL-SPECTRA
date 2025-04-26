from package import *
from hyperparam import *
from model.model import *
from model.training_test import *
from usefull_func import *

# Load the datasets
train_data = datasets.ImageFolder(root=path_train, transform=transform)
test_data = datasets.ImageFolder(root=path_test, transform=transform)
val_data = datasets.ImageFolder(root=path_val, transform=transform)

# Auxiliary variable only used for plotting images
train_data_for_plot = train_data

# Split the dataset: keep only the desired fraction, discard the rest
train_data, _ = random_split(train_data, [int(len(train_data)*(FRACTION)), len(train_data) - int(len(train_data)*(FRACTION))])
test_data, _ = random_split(test_data, [int(len(test_data)*(1)), len(test_data) - int(len(test_data)*(1))])
val_data, _ = random_split(val_data, [int(len(val_data)*(1)), len(val_data) - int(len(val_data)*(1))])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Print dataset sizes for confirmation
print(f'Train set size: {len(train_data)}')
print(f'Validation set size: {len(val_data)}')
print(f'Test set size: {len(test_data)}')

# Model and optimizer setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = CustomCNN(in_channels=in_channels, num_classes=2).to(device)
initialize_weights(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
#scheduler = optim.lr_scheduler.ConstantLR(optimizer)

total_params = count_parameters(model)
print(f"Total number of parameters: {total_params}")

# Lists to store training and validation loss and accuracy
train_losses = []
val_losses = []
train_accurs = []
val_accurs = []

total_real_epoch = 0  # Tracks how many epochs actually run
best_model_state = None
best_val_loss = float('inf')
best_val_accuracy = 0.0
epochs_no_improve = 0

# Start training timer
start_time = time.time()

# Training loop
for epoch in range(1, epochs + 1):
    total_real_epoch += 1
    print(f'\nEpoch {epoch}')
    
    train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch)
    train_losses.append(train_loss)
    train_accurs.append(train_accuracy)

    val_loss, val_accuracy = test(model, device, val_loader)
    val_losses.append(val_loss)
    val_accurs.append(val_accuracy)

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Current learning rate: {current_lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        best_val_accuracy = val_accuracy
        epochs_no_improve = 0
        print(f"New best model found! Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.2f}%")
        torch.save(best_model_state, 'model_weights.pth')
        print(f"Best model weights saved with validation loss: {best_val_loss:.4f} and validation accuracy: {best_val_accuracy:.2f}%")
    else:
        epochs_no_improve += 1
        print("Epoch without improvement: " + str(epochs_no_improve))

    if epochs_no_improve >= patience:
        print("Early stopping")
        break

# End training timer
end_time = time.time()
total_time = end_time - start_time

# Calculate training time
days = total_time // (24 * 3600)
remainder = total_time % (24 * 3600)
hours = remainder // 3600
remainder %= 3600
minutes = remainder // 60
seconds = remainder % 60
print(f'\nTotal training time: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds')

# Load best saved weights
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))

# Final test on the test set
test_loss, test_accuracy = test(model, device, test_loader)
print(f'\nFinal Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.2f}%')
with open('./result/final_test_loss.txt', 'w') as f:
    f.write(f'Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.2f}%')
    f.write(f'\nTotal training time: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds')

# Compute final metrics
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

accuracy, tn, fp, fn, tp, error_rate, precision_pos, precision_neg, recall_pos, recall_neg, fpr_val, fnr_val, f1_pos, f1_neg = calculate_metrics(all_labels, all_preds)

# Save metrics to file
with open('./result/final_metrics.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy:.4f} \n')
    file.write(f'True Negatives: {tn} \n')
    file.write(f'False Positives: {fp} \n')
    file.write(f'False Negatives: {fn} \n')
    file.write(f'True Positives: {tp} \n')
    file.write(f'Error Rate (ER): {error_rate:.4f} \n')
    file.write(f'Precision for Positive Class: {precision_pos:.4f} \n')
    file.write(f'Precision for Negative Class: {precision_neg:.4f} \n')
    file.write(f'True Positive Rate (Recall for Positive Class): {recall_pos:.4f} \n')
    file.write(f'True Negative Rate (Recall for Negative Class): {recall_neg:.4f} \n')
    file.write(f'False Positive Rate (FPR): {fpr_val:.4f} \n')
    file.write(f'False Negative Rate (FNR): {fnr_val:.4f} \n')
    file.write(f'F-Score for Positive Class: {f1_pos:.4f} \n')
    file.write(f'F-Score for Negative Class: {f1_neg:.4f} \n')

print("Final metrics saved in 'final_metrics.txt'.")

# SAVING PLOTS AND DATA --------------------------------------------------------------------------- 1

epochs = np.arange(1, total_real_epoch + 1)
min_test_loss_idx = np.argmin(val_losses)
min_test_loss = val_losses[min_test_loss_idx]
min_test_loss_epoch = min_test_loss_idx + 1

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Training Loss', color='blue')
plt.plot(epochs, val_losses, label='Validation Loss', color='red')
plt.plot(min_test_loss_epoch, min_test_loss, 'ro')
plt.annotate(f'Lowest Validation Loss\n{min_test_loss:.4f}\nEpoch {min_test_loss_epoch}\nAccuracy {best_val_accuracy:.2f}%', 
             xy=(min_test_loss_epoch, min_test_loss), 
             xytext=(min_test_loss_epoch + 1, min_test_loss + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('./result/train_val_loss_plot.png')

with open('./result/train_val_loss_data.txt', 'w') as f:
    f.write('Epoch\tTraining Loss\tValidation Loss\n')
    for epoch, train_loss, val_loss in zip(epochs, train_losses, val_losses):
        f.write(f'{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\n')

print("Plot saved as 'train_val_loss_plot.png' and data saved in 'train_val_loss_data.txt'.")

# SAVING PLOTS AND DATA --------------------------------------------------------------------------- 2

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Training Loss', color='blue', linestyle='--', alpha=0.5)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('./result/train_loss_plot.png')

print("Plot saved as 'train_loss_plot.png'.")

# SAVING PLOTS AND DATA --------------------------------------------------------------------------- 3

max_val_acc_idx = np.argmax(val_accurs)
max_val_acc = val_accurs[max_val_acc_idx]
max_val_acc_epoch = max_val_acc_idx + 1

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accurs, label='Training Accuracy', color='blue')
plt.plot(epochs, val_accurs, label='Validation Accuracy', color='red')
plt.plot(max_val_acc_epoch, max_val_acc, 'ro')
plt.annotate(f'Highest Validation Accuracy\n{max_val_acc:.2f}%\nEpoch {max_val_acc_epoch}', 
             xy=(max_val_acc_epoch, max_val_acc), 
             xytext=(max_val_acc_epoch + 1, max_val_acc - 5),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('./result/train_val_accuracy_plot.png')

with open('./result/train_val_accuracy_data.txt', 'w') as f:
    f.write('Epoch\tTraining Accuracy\tValidation Accuracy\n')
    for epoch, train_acc, val_acc in zip(epochs, train_accurs, val_accurs):
        f.write(f'{epoch}\t{train_acc:.4f}\t{val_acc:.4f}\n')

print("Plot saved as 'train_val_accuracy_plot.png' and data saved in 'train_val_accuracy_data.txt'.")
