# Distributed Training of a CNN using Horovod

import horovod.torch as hvd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

# Initialize Horovod
hvd.init()

# Set up GPU for this process
torch.cuda.set_device(hvd.local_rank())

# Load model
model = models.resnet50(pretrained=True)
model.cuda()

# Wrap model with Horovod DistributedDataParallel
model = hvd.DistributedDataParallel(model)

# Define optimizer and wrap with Horovod DistributedOptimizer
optimizer = optim.SGD(model.parameters(), lr=0.001 * hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

# Load data
train_dataset = datasets.ImageFolder(
    '/dbfs/path/to/train',
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)

# Partition data among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)

# Training loop
for epoch in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0 and hvd.rank() == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

# Save model (on rank 0)
if hvd.rank() == 0:
    torch.save(model.state_dict(), '/dbfs/path/to/model.pth')
