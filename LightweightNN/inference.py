from hyperparam import *
from model.model import *
from model.training_test import *
from usefull_func import *

path_weight = "./model_weights.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model = CustomCNN(in_channels=in_channels, num_classes=2).to(device)

model.load_state_dict(torch.load(path_weight, map_location=device, weights_only=True))

# List of test paths
path_test = [
    "../../dataset_articolo_compatto/dataset_test_selettivo/dataset_test_outlier/dalle2_test",
    "../../dataset_articolo_compatto/dataset_test_selettivo/dataset_test_outlier/dalle3_test",
    "../../dataset_articolo_compatto/dataset_test_selettivo/dataset_test_outlier/firefly_test",
    "../../dataset_articolo_compatto/dataset_test_selettivo/dataset_test_outlier/glide_test",
    "../../dataset_articolo_compatto/dataset_test_selettivo/dataset_test_outlier/midjourney_test",
    "../../dataset_articolo_compatto/dataset_test_selettivo/dataset_test_outlier/stable-diffusion_test",
    "../../dataset_articolo_compatto/dataset_test_selettivo/dataset_test_outlier/dalle-mini_valid_test",
    "../../dataset_articolo_compatto/dataset_test_selettivo/dataset_test_outlier/biggan_test",
    "../../dataset_articolo_compatto/dataset_test_selettivo/dataset_test_outlier/eg3d_test",
    "../../dataset_articolo_compatto/dataset_test_selettivo/dataset_test_outlier/guided-diffusion_test",
    "../../dataset_articolo_compatto/dataset_test_selettivo/dataset_test_outlier/latent-diffusion_test",
    "../../dataset_articolo_compatto/dataset_test_selettivo/dataset_test_outlier/progan_lsun_test",
    "../../dataset_articolo_compatto/dataset_test_selettivo/dataset_test_outlier/stylegan3_test",
    "../../dataset_articolo_compatto/dataset_test_selettivo/dataset_test_outlier/taming-transformers_test",
]

# Run the test for each folder in path_test
for test_path in path_test:
    test_name = os.path.basename(test_path)  # Name of the test folder
    print(f"\nRunning test for: {test_name}...")
    
    # Load the specific test dataset
    test_data = datasets.ImageFolder(root=test_path, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Run the test
    test_loss, test_accuracy = test(model, device, test_loader)
    print(f'\nTest Loss for {test_name}: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # Collect predictions and labels
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate all required metrics
    accuracy, tn, fp, fn, tp, error_rate, precision_pos, precision_neg, recall_pos, recall_neg, fpr_val, fnr_val, f1_pos, f1_neg = calculate_metrics(all_labels, all_preds)

    # Create a directory for the specific test results, if it doesn't exist
    results_dir = f'./result/{test_name}'
    os.makedirs(results_dir, exist_ok=True)

    # Save the metrics to a specific file
    results_file_path = os.path.join(results_dir, 'metrics.txt')
    with open(results_file_path, 'w') as file:
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

    print(f"Metrics for {test_name} saved in '{results_file_path}'.")
