from package import *

# Main parameters
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-2  # if 0, there is no regularization; otherwise, values around 1e-4 or 1e-2

epochs = 100

image_width = 222
image_height = 222
in_channels = 3

# Use only a fraction of the dataset to reduce size; set to 1 to use the entire dataset
FRACTION = 1

patience = 15  # Number of epochs to wait before stopping training (typically between 5 and 10, higher for testing)

path_train = '../../dataset_articolo_new/dataset/train'
path_test = '../../dataset_articolo_new/dataset/test'
path_val = '../../dataset_articolo_new/dataset/val'

# Transformations to normalize the data
transform = transforms.Compose([
    transforms.Resize((image_width, image_height)),
    # transforms.Grayscale(), # uncomment if working with in_channel=1, otherwise keep commented for in_channel=3
    transforms.ToTensor(),
])
