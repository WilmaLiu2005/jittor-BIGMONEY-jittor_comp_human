import jittor as jt
import numpy as np
import os
import argparse
import time
import random
import matplotlib.pyplot as plt

from jittor import nn
from jittor import optim

from dataset.dataset import get_dataloader, transform
from dataset.format import id_to_name
from dataset.sampler import SamplerMix
from models.skin import create_model

from dataset.exporter import Exporter

# Set Jittor flags
jt.flags.use_cuda = 1

def plot_training_progress(train_losses_mse, train_losses_l1, val_losses_mse, val_losses_l1, epochs_logged, save_path):
    """
    Plot training progress and save to file
    
    Args:
        train_losses_mse: List of training MSE losses
        train_losses_l1: List of training L1 losses
        val_losses_mse: List of validation MSE losses
        val_losses_l1: List of validation L1 losses
        epochs_logged: List of epochs when validation was performed
        save_path: Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training losses
    ax1.plot(range(1, len(train_losses_mse) + 1), train_losses_mse, 'b-', label='Train MSE Loss', linewidth=2)
    ax1.plot(range(1, len(train_losses_l1) + 1), train_losses_l1, 'g-', label='Train L1 Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')
    
    # Plot validation losses
    if val_losses_mse and val_losses_l1:
        ax2.plot(epochs_logged, val_losses_mse, 'r-', label='Val MSE Loss', linewidth=2)
        ax2.plot(epochs_logged, val_losses_l1, 'orange', label='Val L1 Loss', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Validation Losses')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_yscale('log')
    
    # Plot MSE losses comparison
    if val_losses_mse:
        train_mse_at_val = [train_losses_mse[epoch-1] for epoch in epochs_logged]
        ax3.plot(epochs_logged, train_mse_at_val, 'b-', label='Train MSE', linewidth=2)
        ax3.plot(epochs_logged, val_losses_mse, 'r-', label='Val MSE', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('MSE Loss')
        ax3.set_title('MSE Loss Comparison')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_yscale('log')
    
    # Plot L1 losses comparison
    if val_losses_l1:
        train_l1_at_val = [train_losses_l1[epoch-1] for epoch in epochs_logged]
        ax4.plot(epochs_logged, train_l1_at_val, 'g-', label='Train L1', linewidth=2)
        ax4.plot(epochs_logged, val_losses_l1, 'orange', label='Val L1', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('L1 Loss')
        ax4.set_title('L1 Loss Comparison')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    
    def log_message(message):
        """Helper function to log messages to file and print to console"""
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    # Initialize lists to store training history for plotting
    train_losses_mse = []
    train_losses_l1 = []
    val_losses_mse = []
    val_losses_l1 = []
    epochs_logged = []
    
    # Log training parameters
    log_message(f"Starting training with parameters: {args}")
    
    # Create model
    model = create_model(
        model_name=args.model_name,
    )
    
    # Load pre-trained model if specified
    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)
    
    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Create loss function
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    
    # Create dataloaders
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=SamplerMix(num_samples=1024, vertex_samples=512),
        transform=transform,
    )
    
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=SamplerMix(num_samples=1024, vertex_samples=512),
            transform=transform,
        )
    else:
        val_loader = None
    
    # Training loop
    best_loss = 99999999
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss_mse = 0.0
        train_loss_l1 = 0.0
        
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            # Get data and labels
            vertices, joints, skin = data['vertices'], data['joints'], data['skin']

            vertices: jt.Var
            joints: jt.Var
            skin: jt.Var
            outputs = model(vertices, joints)
            loss_mse = criterion_mse(outputs, skin)
            loss_l1 = criterion_l1(outputs, skin)
            loss = loss_mse + loss_l1
            
            # Backward pass and optimize
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            # Calculate statistics
            train_loss_mse += loss_mse.item()
            train_loss_l1 += loss_l1.item()
            
            # Print progress
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss mse: {loss_mse.item():.4f} Loss l1: {loss_l1.item():.4f}")
        
        # Calculate epoch statistics
        train_loss_mse /= len(train_loader)
        train_loss_l1 /= len(train_loader)
        train_losses_mse.append(train_loss_mse)
        train_losses_l1.append(train_loss_l1)
        epoch_time = time.time() - start_time
        
        log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                   f"Train Loss mse: {train_loss_mse:.4f} "
                   f"Train Loss l1: {train_loss_l1:.4f} "
                   f"Time: {epoch_time:.2f}s "
                   f"LR: {optimizer.lr:.6f}")

        # Validation phase
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss_mse = 0.0
            val_loss_l1 = 0.0
            
            show_id = np.random.randint(0, len(val_loader))
            for batch_idx, data in enumerate(val_loader):
                # Get data and labels
                vertices, joints, skin = data['vertices'], data['joints'], data['skin']
                
                # Forward pass
                outputs = model(vertices, joints)
                loss_mse = criterion_mse(outputs, skin)
                loss_l1 = criterion_l1(outputs, skin)
                
                # export render results(which is slow, so you can turn it off)
                if batch_idx == show_id:
                    exporter = Exporter()
                    for i in id_to_name:
                        name = id_to_name[i]
                        # export every joint's corresponding skinning
                        exporter._render_skin(path=f"tmp/skin/epoch_{epoch}/{name}_ref.png",vertices=vertices.numpy()[0], skin=skin.numpy()[0, :, i], joint=joints[0, i])
                        exporter._render_skin(path=f"tmp/skin/epoch_{epoch}/{name}_pred.png",vertices=vertices.numpy()[0], skin=outputs.numpy()[0, :, i], joint=joints[0, i])

                val_loss_mse += loss_mse.item()
                val_loss_l1 += loss_l1.item()
            
            # Calculate validation statistics
            val_loss_mse /= len(val_loader)
            val_loss_l1 /= len(val_loader)
            
            # Store validation metrics for plotting
            val_losses_mse.append(val_loss_mse)
            val_losses_l1.append(val_loss_l1)
            epochs_logged.append(epoch + 1)
            
            log_message(f"Validation Loss: mse: {val_loss_mse:.4f} l1: {val_loss_l1:.4f}")
            
            # Save best model
            if val_loss_l1 < best_loss:
                best_loss = val_loss_l1
                model_path = os.path.join(args.output_dir, 'best_model.pkl')
                model.save(model_path)
                log_message(f"Saved best model with l1 loss {best_loss:.4f} to {model_path}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save(checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")
        
        # Update plot during training if enabled
        if args.plot_training and (epoch + 1) % args.plot_freq == 0:
            plot_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.plot_name)
            plot_training_progress(train_losses_mse, train_losses_l1, val_losses_mse, val_losses_l1, epochs_logged, plot_save_path)
            log_message(f"Updated training plot saved to {plot_save_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"Training completed. Saved final model to {final_model_path}")
    
    # Generate final training plot
    if args.plot_training:
        plot_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.plot_name)
        plot_training_progress(train_losses_mse, train_losses_l1, val_losses_mse, val_losses_l1, epochs_logged, plot_save_path)
        log_message(f"Final training plot saved to {plot_save_path}")
    
    return model, best_loss

def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description='Train a point cloud model')
    
    # Dataset parameters
    parser.add_argument('--train_data_list', type=str, required=True,
                        help='Path to the training data list file')
    parser.add_argument('--val_data_list', type=str, default='',
                        help='Path to the validation data list file')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for the data files')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='pct',
                        choices=['pct', 'pct2', 'custom_pct', 'skeleton'],
                        help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'enhanced'],
                        help='Model type for skeleton model')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Path to pretrained model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output/skin',
                        help='Directory to save output files')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save frequency')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Validation frequency')
    
    # Plotting parameters
    parser.add_argument('--plot_training', action='store_true', default=True,
                        help='Enable training progress plotting')
    parser.add_argument('--plot_name', type=str, default='skin.png',
                        help='Name of the plot file to save')
    parser.add_argument('--plot_freq', type=int, default=10,
                        help='Frequency to update the plot during training')
    
    args = parser.parse_args()
    
    # Start training
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(3407)
    main()