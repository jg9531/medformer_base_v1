import torch
from tqdm import tqdm

def train(model, train_loader, criterion, optimizer, task, device):
    model.train()
    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

import torch
from medmnist import Evaluator

def evaluate(model, data_loader, data_flag, split, device):
    model.eval()
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for inputs, _ in tqdm(data_loader, desc=f"Evaluating [{split}]"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.softmax(dim=-1)
            y_score = torch.cat((y_score, outputs), dim=0)

    y_score = y_score.cpu().detach().numpy()
    evaluator = Evaluator(data_flag, split, size=224)
    metrics = evaluator.evaluate(y_score)
    print('%s  auc: %.3f  acc: %.3f' % (split, *metrics))
