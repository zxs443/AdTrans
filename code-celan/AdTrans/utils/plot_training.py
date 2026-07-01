import os
import matplotlib.pyplot as plt
from datetime import datetime
from utils.visualization_config import VISUALIZATION_CONFIG


plt.rcParams['font.family'] = VISUALIZATION_CONFIG['font_family']

def plot_training_curves(train_losses: list, val_losses: list, save_dir: str, descriptor_type: str):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(VISUALIZATION_CONFIG['figsize'][0], VISUALIZATION_CONFIG['figsize'][1] * 0.7), dpi=VISUALIZATION_CONFIG['dpi'])
    plt.plot(epochs, train_losses, label='Train Loss', color=VISUALIZATION_CONFIG['palette']['Train'])
    plt.plot(epochs, val_losses, label='Validation Loss', color=VISUALIZATION_CONFIG['palette']['Test'])
    plt.title(f'Training and Validation Loss - {descriptor_type}', fontsize=VISUALIZATION_CONFIG['font_size_title'])
    plt.xlabel('Epoch', fontsize=VISUALIZATION_CONFIG['font_size_label'])
    plt.ylabel('Loss (MSE)', fontsize=VISUALIZATION_CONFIG['font_size_label'])
    plt.legend(fontsize=VISUALIZATION_CONFIG['font_size_legend'])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, f'training_loss_curves_{descriptor_type}_{timestamp}.png')
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=VISUALIZATION_CONFIG['dpi'])
    plt.close()
    print(f"Training loss curve plot saved to: {save_path}")

def plot_learning_rate(learning_rates: list, save_dir: str, descriptor_type: str):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    epochs = range(1, len(learning_rates) + 1)

    plt.figure(figsize=(VISUALIZATION_CONFIG['figsize'][0], VISUALIZATION_CONFIG['figsize'][1] * 0.7), dpi=VISUALIZATION_CONFIG['dpi'])
    plt.plot(epochs, learning_rates, label='Learning Rate', color='purple')
    plt.title(f'Learning Rate Schedule - {descriptor_type}', fontsize=VISUALIZATION_CONFIG['font_size_title'])
    plt.xlabel('Epoch', fontsize=VISUALIZATION_CONFIG['font_size_label'])
    plt.ylabel('Learning Rate', fontsize=VISUALIZATION_CONFIG['font_size_label'])
    plt.yscale('log') 
    plt.legend(fontsize=VISUALIZATION_CONFIG['font_size_legend'])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, f'learning_rate_curve_{descriptor_type}_{timestamp}.png')
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=VISUALIZATION_CONFIG['dpi'])
    plt.close()
    print(f"Learning rate curve plot saved to: {save_path}")
