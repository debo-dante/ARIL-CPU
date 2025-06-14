import os
import gc
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm

from models.transformer_csi import TransformerCSI

# Configuration parameters
batch_size = 128
num_epochs = 200

# Create directories if they don't exist
data_dir = 'data'
weights_dir = 'weights'
results_dir = 'result'

for directory in [data_dir, weights_dir, results_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load data
data_amp = sio.loadmat(os.path.join(data_dir, 'train_data_split_amp.mat'))
train_data_amp = data_amp['train_data']
train_data = train_data_amp
# data_pha = sio.loadmat(os.path.join(data_dir, 'train_data_split_pha.mat'))
# train_data_pha = data_pha['train_data']
# train_data = np.concatenate((train_data_amp,train_data_pha),1)

train_activity_label = data_amp['train_activity_label']
train_location_label = data_amp['train_location_label']
train_label = np.concatenate((train_activity_label, train_location_label), 1)

num_train_instances = len(train_data)
print(f"Number of training instances: {num_train_instances}")

train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
train_label = torch.from_numpy(train_label).type(torch.LongTensor)
# train_data = train_data.view(num_train_instances, 1, -1)
# train_label = train_label.view(num_train_instances, 2)

train_dataset = TensorDataset(train_data, train_label)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

data_amp = sio.loadmat(os.path.join(data_dir, 'test_data_split_amp.mat'))
test_data_amp = data_amp['test_data']
test_data = test_data_amp
# data_pha = sio.loadmat(os.path.join(data_dir, 'test_data_split_pha.mat'))
# test_data_pha = data_pha['test_data']
# test_data = np.concatenate((test_data_amp,test_data_pha), 1)

test_activity_label = data_amp['test_activity_label']
test_location_label = data_amp['test_location_label']
test_label = np.concatenate((test_activity_label, test_location_label), 1)

num_test_instances = len(test_data)
print(f"Number of test instances: {num_test_instances}")

test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label).type(torch.LongTensor)
# test_data = test_data.view(num_test_instances, 1, -1)
# test_label = test_label.view(num_test_instances, 2)

test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# Initialize model
print("Initializing TransformerCSI model...")
aplnet = TransformerCSI(
    input_dim=52,    # Number of input features
    hidden_dim=128,  # Size of hidden dimension
    num_heads=4,     # Number of attention heads
    num_layers=3,    # Number of transformer layers
    num_classes_act=6,  # Number of activity classes
    num_classes_loc=16  # Number of location classes
)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(size_average=False)
optimizer = torch.optim.Adam(aplnet.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[10, 20, 30, 40, 60, 70, 80, 90, 100, 110, 120, 130,
                140, 150, 160, 170, 180, 190, 200, 250, 300],
    gamma=0.5
)

# Print model summary
print(f"Model parameters: {sum(p.numel() for p in aplnet.parameters() if p.requires_grad):,}")
train_loss_act = np.zeros([num_epochs, 1])
train_loss_loc = np.zeros([num_epochs, 1])
test_loss_act = np.zeros([num_epochs, 1])
test_loss_loc = np.zeros([num_epochs, 1])
train_acc_act = np.zeros([num_epochs, 1])
train_acc_loc = np.zeros([num_epochs, 1])
test_acc_act = np.zeros([num_epochs, 1])
test_acc_loc = np.zeros([num_epochs, 1])

print("Starting training...")
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    print(f'Epoch: {epoch+1}/{num_epochs}')
    
    # Clear memory
    if hasattr(torch, 'cuda'):
        torch.cuda.empty_cache()
    gc.collect()
    
    # Training phase
    aplnet.train()
    scheduler.step()
    loss_x = 0
    loss_y = 0
    
    # Enhanced progress bar
    progress_bar = tqdm(train_data_loader, desc="Training", unit="batch")
    for i, (samples, labels) in enumerate(progress_bar):
        samplesV = Variable(samples)
        labels_act = labels[:, 0].squeeze()
        labels_loc = labels[:, 1].squeeze()
        labelsV_act = Variable(labels_act)
        labelsV_loc = Variable(labels_loc)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        predict_label_act, predict_label_loc = aplnet(samplesV)
        
        loss_act = criterion(predict_label_act, labelsV_act)
        loss_loc = criterion(predict_label_loc, labelsV_loc)

        loss = loss_act + loss_loc
        loss.backward()
        optimizer.step()

        # Update running loss values
        loss_x += loss_act.item()
        loss_y += loss_loc.item()
        
        # Update progress bar with metrics
        progress_bar.set_postfix({
            'act_loss': loss_act.item() / len(labels),
            'loc_loss': loss_loc.item() / len(labels),
            'lr': optimizer.param_groups[0]['lr']
        })

    train_loss_act[epoch] = loss_x / num_train_instances
    train_loss_loc[epoch] = loss_y / num_train_instances

    aplnet.eval()
    # loss_x = 0
    correct_train_act = 0
    correct_train_loc = 0
    for i, (samples, labels) in enumerate(train_data_loader):
        with torch.no_grad():
            samplesV = Variable(samples)  # Removed .cuda() call
            labels = labels.squeeze()

            labels_act = labels[:, 0].squeeze()
            labels_loc = labels[:, 1].squeeze()
            labelsV_act = Variable(labels_act)  # Removed .cuda() call
            labelsV_loc = Variable(labels_loc)  # Removed .cuda() call

            predict_label_act, predict_label_loc = aplnet(samplesV)

            prediction = predict_label_loc.data.max(1)[1]
            correct_train_loc += prediction.eq(labelsV_loc.data.long()).sum()

            prediction = predict_label_act.data.max(1)[1]
            correct_train_act += prediction.eq(labelsV_act.data.long()).sum()

            loss_act = criterion(predict_label_act, labelsV_act)
            loss_loc = criterion(predict_label_loc, labelsV_loc)
            # loss_x += loss.item()

    print("Activity Training accuracy:", (100 * float(correct_train_act) / num_train_instances))
    print("Location Training accuracy:", (100 * float(correct_train_loc) / num_train_instances))

    # train_loss[epoch] = loss_x / num_train_instances
    train_acc_act[epoch] = 100 * float(correct_train_act) / num_train_instances
    train_acc_loc[epoch] = 100 * float(correct_train_loc) / num_train_instances


    trainacc_act = str(100 * float(correct_train_act) / num_train_instances)[0:6]
    trainacc_loc = str(100 * float(correct_train_loc) / num_train_instances)[0:6]

    loss_x = 0
    loss_y = 0
    correct_test_act = 0
    correct_test_loc = 0
    for i, (samples, labels) in enumerate(test_data_loader):
        with torch.no_grad():
            samplesV = Variable(samples)  # Removed .cuda() call
            labels_act = labels[:, 0].squeeze()
            labels_loc = labels[:, 1].squeeze()
            labelsV_act = Variable(labels_act)  # Removed .cuda() call
            labelsV_loc = Variable(labels_loc)  # Removed .cuda() call

        predict_label_act, predict_label_loc = aplnet(samplesV)
        prediction = predict_label_act.data.max(1)[1]
        correct_test_act += prediction.eq(labelsV_act.data.long()).sum()

        prediction = predict_label_loc.data.max(1)[1]
        correct_test_loc += prediction.eq(labelsV_loc.data.long()).sum()

        loss_act = criterion(predict_label_act, labelsV_act)
        loss_loc = criterion(predict_label_loc, labelsV_loc)
        loss_x += loss_act.item()
        loss_y += loss_loc.item()

    print("Activity Test accuracy:", (100 * float(correct_test_act) / num_test_instances))
    print("Location Test accuracy:", (100 * float(correct_test_loc) / num_test_instances))

    test_loss_act[epoch] = loss_x / num_test_instances
    test_acc_act[epoch] = 100 * float(correct_test_act) / num_test_instances

    test_loss_loc[epoch] = loss_y / num_test_instances
    test_acc_loc[epoch] = 100 * float(correct_test_loc) / num_test_instances

    testacc_act = str(100 * float(correct_test_act) / num_test_instances)[0:6]
    testacc_loc = str(100 * float(correct_test_loc) / num_test_instances)[0:6]

    # Save best model based on test accuracy
    if epoch == 0:
        temp_test = correct_test_act
        temp_train = correct_train_act
    elif correct_test_act > temp_test:
        model_filename = os.path.join(
            weights_dir, 
            f'net_epoch{epoch}_TrainAct{trainacc_act}_TestAct{testacc_act}_TrainLoc{trainacc_loc}_TestLoc{testacc_loc}.pkl'
        )
        torch.save(aplnet, model_filename)
        print(f"New best model saved to {model_filename}")

        temp_test = correct_test_act
        temp_train = correct_train_act
    
    # Print epoch summary
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

# Final memory cleanup
if hasattr(torch, 'cuda'):
    torch.cuda.empty_cache()
gc.collect()

# Calculate final metrics
train_acc_pct = 100 * float(temp_train) / num_train_instances
test_acc_pct = 100 * float(temp_test) / num_test_instances
train_acc_str = f"{train_acc_pct:.2f}"
test_acc_str = f"{test_acc_pct:.2f}"

print(f"Best test accuracy: {test_acc_str}%")

# Save learning curves
timestamp = time.strftime("%Y%m%d_%H%M%S")
results_prefix = os.path.join(results_dir, f'results_{timestamp}')

# Save all results in a more organized way
results = {
    f"{results_prefix}_TrainLossAct.mat": {'train_loss': train_loss_act},
    f"{results_prefix}_TestLossAct.mat": {'test_loss': test_loss_act},
    f"{results_prefix}_TrainLossLoc.mat": {'train_loss': train_loss_loc},
    f"{results_prefix}_TestLossLoc.mat": {'test_loss': test_loss_loc},
    f"{results_prefix}_TrainAccAct.mat": {'train_acc': train_acc_act},
    f"{results_prefix}_TestAccAct.mat": {'test_acc': test_acc_act},
    f"{results_prefix}_TrainAccLoc.mat": {'train_acc': train_acc_loc},
    f"{results_prefix}_TestAccLoc.mat": {'test_acc': test_acc_loc}
}

print("Saving results...")
for filename, data in results.items():
    sio.savemat(filename, data)
    
print(f"Training completed. Results saved to {results_dir}")

# Plot training curves
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(train_loss_act)
plt.title('Activity Training Loss')

plt.subplot(2, 2, 2)
plt.plot(test_loss_act)
plt.title('Activity Testing Loss')

plt.subplot(2, 2, 3)
plt.plot(train_acc_act)
plt.title('Activity Training Accuracy')

plt.subplot(2, 2, 4)
plt.plot(test_acc_act)
plt.title('Activity Testing Accuracy')

plt.tight_layout()
plt.savefig(f"{results_prefix}_activity_plots.png")
print(f"Training plots saved to {results_prefix}_activity_plots.png")
