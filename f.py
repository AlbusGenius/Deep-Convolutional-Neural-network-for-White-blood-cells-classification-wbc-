import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Improved data transformations
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # Reduced size for better training
    transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
    transforms.RandomRotation(degrees=15),    # Data augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet standards
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Apply different transforms to train and test
dataset = datasets.ImageFolder(root='path_to_the_dataset', transform=transform_train)

train_len = int(0.8 * len(dataset))
test_len = len(dataset) - train_len
train_set, test_set = random_split(dataset, [train_len, test_len])

# Apply test transform to test set
test_set.dataset = datasets.ImageFolder(root='path_to_the_datase', transform=transform_test)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)  # Reduced batch size
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ImprovedCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize model
model = ImprovedCNN(num_classes=5).to(device)

#  optimizer and loss function
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for regularization
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
            })
        
        # Calculate training metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / total_predictions
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_epoch_loss = val_running_loss / len(test_loader.dataset)
        val_epoch_acc = val_correct / val_total
        
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        
        # Step the scheduler
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 50)
    
    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Evaluating')
        for images, labels in test_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            test_pbar.set_postfix({'Acc': f'{100 * correct / total:.2f}%'})
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f} ({100 * accuracy:.2f}%)')
    
    return accuracy, all_predictions, all_labels

def plot_confusion_matrix(true_labels, predictions, class_names, title='Confusion Matrix'):
    """
    Plot confusion matrix using seaborn
    """
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return cm

def analyze_model_performance(true_labels, predictions, class_names):
    """
    Comprehensive performance analysis including per-class metrics
    """
    print("\n" + "="*80)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 50)
    report = classification_report(true_labels, predictions, 
                                   target_names=class_names if class_names else None)
    print(report)
    
    # Per-class accuracy
    cm = confusion_matrix(true_labels, predictions)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    print("\nPer-Class Accuracy:")
    print("-" * 50)
    if class_names:
        for i, class_name in enumerate(class_names):
            print(f"{class_name}: {per_class_accuracy[i]:.4f} ({per_class_accuracy[i]*100:.2f}%)")
    else:
        for i in range(len(per_class_accuracy)):
            print(f"Class {i}: {per_class_accuracy[i]:.4f} ({per_class_accuracy[i]*100:.2f}%)")
    
    # Most confused pairs
    print("\nMost Confused Class Pairs:")
    print("-" * 50)
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    
    
    flat_indices = np.argpartition(cm_no_diag.ravel(), -3)[-3:]
    pairs = [(idx // cm.shape[1], idx % cm.shape[1]) for idx in flat_indices]
    
    for true_idx, pred_idx in sorted(pairs, key=lambda x: cm[x[0], x[1]], reverse=True):
        if cm[true_idx, pred_idx] > 0:
            if class_names:
                print(f"{class_names[true_idx]} -> {class_names[pred_idx]}: {cm[true_idx, pred_idx]} samples")
            else:
                print(f"Class {true_idx} -> Class {pred_idx}: {cm[true_idx, pred_idx]} samples")

# Feature Map Visualization Functions (keeping all original visualization functions)
def get_feature_maps(model, input_tensor, layer_names=None):
    """
    Extract feature maps from specified layers during forward pass
    """
    feature_maps = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach()
        return hook
    
    # Register hooks for different layers
    if layer_names is None:
        # Default layers to visualize
        layer_names = [
            'features.0',   # First conv layer
            'features.8',   # Second conv block first conv
            'features.16',  # Third conv block first conv
            'features.24',  # Fourth conv block first conv
        ]
    
    # Register hooks
    for name in layer_names:
        layer = model
        for attr in name.split('.'):
            layer = getattr(layer, attr)
        hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return feature_maps

def visualize_feature_maps(feature_maps, input_image, class_names=None, predicted_class=None, true_class=None):
    """
    Visualize feature maps from different layers
    """
    # Convert input image back to displayable format
    if input_image.dim() == 4:
        input_image = input_image.squeeze(0)
    
    # Denormalize the input image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    input_img_denorm = input_image * std + mean
    input_img_denorm = torch.clamp(input_img_denorm, 0, 1)
    
    # Create figure
    num_layers = len(feature_maps)
    fig = plt.figure(figsize=(20, 4 * num_layers + 2))
    
    # Show original image
    ax_orig = plt.subplot(num_layers + 1, 8, 1)
    plt.imshow(input_img_denorm.permute(1, 2, 0))
    title = "Original Image"
    if class_names and predicted_class is not None and true_class is not None:
        title += f"\nTrue: {class_names[true_class]}\nPred: {class_names[predicted_class]}"
    ax_orig.set_title(title, fontsize=10)
    ax_orig.axis('off')
    
    # Show feature maps for each layer
    for layer_idx, (layer_name, feature_map) in enumerate(feature_maps.items()):
        if feature_map.dim() == 4:
            feature_map = feature_map.squeeze(0)  # Remove batch dimension
        
        num_channels = feature_map.shape[0]
        
        # Show first 8 feature maps for this layer
        for i in range(min(8, num_channels)):
            ax = plt.subplot(num_layers + 1, 8, (layer_idx + 1) * 8 + i + 1)
            
            # Get single feature map
            fmap = feature_map[i].cpu().numpy()
            
            # Display feature map
            im = ax.imshow(fmap, cmap='viridis')
            ax.set_title(f'{layer_name}\nChannel {i}', fontsize=8)
            ax.axis('off')
            
            # Add colorbar for first feature map of each layer
            if i == 0:
                plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.show()

def visualize_filters(model, layer_name='features.0', max_filters=16):
    """
    Visualize the actual learned filters/kernels
    """
    # Get the specified layer
    layer = model
    for attr in layer_name.split('.'):
        layer = getattr(layer, attr)
    
    if hasattr(layer, 'weight'):
        filters = layer.weight.data.cpu()
        
        # Create figure
        num_filters = min(max_filters, filters.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(num_filters):
            filter_weights = filters[i]
            
            # If filter has multiple input channels, show them separately or combined
            if filter_weights.shape[0] == 3:  # RGB filter
                # Normalize for display
                filter_normalized = (filter_weights - filter_weights.min()) / (filter_weights.max() - filter_weights.min() + 1e-8)
                axes[i].imshow(filter_normalized.permute(1, 2, 0))
            else:
                # For single channel or averaged
                if filter_weights.shape[0] > 1:
                    filter_avg = filter_weights.mean(dim=0)
                else:
                    filter_avg = filter_weights[0]
                axes[i].imshow(filter_avg, cmap='gray')
            
            axes[i].set_title(f'Filter {i}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_filters, 16):
            axes[i].axis('off')
        
        plt.suptitle(f'Learned Filters from Layer: {layer_name}', fontsize=16)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Layer {layer_name} does not have weights to visualize")

def analyze_feature_activation_patterns(feature_maps, class_names=None, true_class=None):
    """
    Analyze activation patterns across different layers
    """
    print("\n" + "="*80)
    print("FEATURE ACTIVATION ANALYSIS")
    print("="*80)
    
    for layer_name, feature_map in feature_maps.items():
        if feature_map.dim() == 4:
            feature_map = feature_map.squeeze(0)
        
        # Calculate statistics
        mean_activation = feature_map.mean().item()
        max_activation = feature_map.max().item()
        min_activation = feature_map.min().item()
        std_activation = feature_map.std().item()
        
        # Calculate sparsity (percentage of near-zero activations)
        threshold = 0.1 * max_activation if max_activation > 0 else 0.01
        sparsity = (feature_map.abs() < threshold).float().mean().item() * 100
        
        print(f"\n{layer_name}:")
        print(f"  Shape: {list(feature_map.shape)}")
        print(f"  Mean activation: {mean_activation:.4f}")
        print(f"  Max activation: {max_activation:.4f}")
        print(f"  Std activation: {std_activation:.4f}")
        print(f"  Sparsity: {sparsity:.1f}% (activations < {threshold:.3f})")
        
        # Find most active channels
        channel_means = feature_map.view(feature_map.shape[0], -1).mean(dim=1)
        top_channels = torch.topk(channel_means, k=min(3, len(channel_means)))
        print(f"  Most active channels: {top_channels.indices.tolist()} (mean activations: {[f'{v:.4f}' for v in top_channels.values.tolist()]})")

# Train the model
EPOCHS = 20
print("Starting training...")
train_losses, train_accuracies, val_losses, val_accuracies = train_model(
    model, train_loader, test_loader, criterion, optimizer, scheduler, EPOCHS
)

# Evaluate the model with comprehensive analysis
print("\nEvaluating model...")
test_accuracy, predictions, true_labels = evaluate_model(model, test_loader)

# Get class names from dataset
class_names = dataset.classes if hasattr(dataset, 'classes') else None
print(f"\nClasses detected: {class_names}")

# Plot confusion matrix
print("\nGenerating confusion matrix...")
cm = plot_confusion_matrix(true_labels, predictions, class_names, 
                          'White Blood Cell Classification Results')

# Comprehensive performance analysis
analyze_model_performance(true_labels, predictions, class_names)

# Feature Map Visualization
print("\n" + "="*80)
print("FEATURE MAP VISUALIZATION")
print("="*80)

# Get a few test samples for visualization
model.eval()
sample_indices = [0, 1, 2]  # Visualize first 3 test samples

for idx in sample_indices:
    print(f"\nVisualizing sample {idx + 1}:")
    
    # Get sample
    if hasattr(test_set, 'dataset'):
        # Handle random_split case
        actual_idx = test_set.indices[idx]
        image, label = test_set.dataset[actual_idx]
    else:
        image, label = test_set[idx]
    
    # Add batch dimension
    input_tensor = image.unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()
    
    # Get feature maps
    feature_maps = get_feature_maps(model, input_tensor)
    
    # Visualize
    visualize_feature_maps(feature_maps, input_tensor.cpu(), class_names, predicted, label)
    
    # Analyze activation patterns
    analyze_feature_activation_patterns(feature_maps, class_names, label)
    
    print("-" * 50)

# Visualize learned filters
print("\nVisualizing learned filters from first layer:")
visualize_filters(model, 'features.0', max_filters=16)

print("\nVisualizing learned filters from second layer:")
visualize_filters(model, 'features.8', max_filters=16)

# Plot training curves
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot([acc * 100 for acc in train_accuracies], label='Train Accuracy')
plt.plot([acc * 100 for acc in val_accuracies], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.subplot(1, 3, 3)
epochs_run = len(train_losses)
plt.plot(range(1, epochs_run + 1), [acc * 100 for acc in val_accuracies])
plt.title('Validation Accuracy Progress')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.axhline(y=90, color='r', linestyle='--', label='Target (90%)')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
torch.save(model.state_dict(), 'improved_wbc_classifier.pth')
print("\nModel saved as 'improved_wbc_classifier.pth'")