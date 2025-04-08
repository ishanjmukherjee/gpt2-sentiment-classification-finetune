# %%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the SST-2 dataset
sst2_dataset = load_dataset("glue", "sst2")
train_dataset = sst2_dataset["train"]
validation_dataset = sst2_dataset["validation"]
test_dataset = sst2_dataset["test"]

# %%

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Add padding token to tokenizer
tokenizer.pad_token = tokenizer.eos_token

print(tokenizer)

# %%

# Define max sequence length
max_length = 128

# Custom dataset class to format examples properly
class GPT2ClassificationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Format examples: "Review: {text} Sentiment:"
        text = f"Review: {self.dataset[idx]['sentence']} Sentiment:"
        label = self.dataset[idx]['label']

        # Tokenize the text
        encoding = self.tokenizer(text,
                                 return_tensors='pt',
                                 max_length=self.max_length,
                                 padding='max_length',
                                 truncation=True)

        # Extract input_ids and attention_mask
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create custom datasets
train_data = GPT2ClassificationDataset(train_dataset, tokenizer, max_length)
val_data = GPT2ClassificationDataset(validation_dataset, tokenizer, max_length)
test_data = GPT2ClassificationDataset(test_dataset, tokenizer, max_length)

# Create data loaders
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)


# %%

# Define the GPT-2 with classification head model
class GPT2ForSentimentClassification(nn.Module):
    def __init__(self, num_classes=2, unfreeze_layers=2):
        super(GPT2ForSentimentClassification, self).__init__()
        # Load pre-trained GPT-2 model
        self.gpt2 = GPT2Model.from_pretrained('gpt2')

        # Get GPT-2 configuration
        config = self.gpt2.config

        # Add classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.n_embd, num_classes)
        )

        # Freeze most of the GPT-2 layers
        for param in self.gpt2.parameters():
            param.requires_grad = False

        # Unfreeze only the last few transformer blocks
        for i, block in enumerate(self.gpt2.h[-unfreeze_layers:]):
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        # Get GPT-2 outputs
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)

        # Use the last hidden state of the last token for classification
        last_hidden_state = outputs.last_hidden_state
        batch_size = last_hidden_state.shape[0]

        # Get the last token representation for each sequence
        # by using the attention mask to identify the last token
        last_token_indices = attention_mask.sum(dim=1) - 1
        last_token_hidden = torch.stack([
            last_hidden_state[i, last_idx, :]
            for i, last_idx in enumerate(last_token_indices)
        ])

        # Pass through the classification head
        logits = self.classifier(last_token_hidden)

        return logits

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = GPT2ForSentimentClassification(num_classes=2, unfreeze_layers=2).to(device)

# %%

# Set up training parameters
learning_rate = 5e-5
num_epochs = 3
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    best_accuracy = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Store loss and predictions
            train_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            train_preds.extend(preds.cpu().tolist())
            train_labels.extend(labels.cpu().tolist())

        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_accuracy = accuracy_score(train_labels, train_preds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                # Store loss and predictions
                val_loss += loss.item()
                _, preds = torch.max(logits, dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)

        # Print metrics
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "gpt2_sentiment_classifier.pt")
            print(f"  New best model saved with accuracy: {best_accuracy:.4f}")

    return model

# Train the model
trained_model = train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device)

"""
Kernel output:

Epoch 1/3 [Training]: 100%|██████████| 4210/4210 [01:44<00:00, 40.26it/s]
Epoch 1/3 [Validation]: 100%|██████████| 55/55 [00:01<00:00, 51.32it/s]
Epoch 1/3:
  Train Loss: 0.3254, Train Accuracy: 0.8556
  Val Loss: 0.2799, Val Accuracy: 0.8945
  New best model saved with accuracy: 0.8945
Epoch 2/3 [Training]: 100%|██████████| 4210/4210 [01:43<00:00, 40.66it/s]
Epoch 2/3 [Validation]: 100%|██████████| 55/55 [00:01<00:00, 53.87it/s]
Epoch 2/3:
  Train Loss: 0.2541, Train Accuracy: 0.8948
  Val Loss: 0.2637, Val Accuracy: 0.8911
Epoch 3/3 [Training]: 100%|██████████| 4210/4210 [01:43<00:00, 40.62it/s]
Epoch 3/3 [Validation]: 100%|██████████| 55/55 [00:01<00:00, 53.54it/s]
Epoch 3/3:
  Train Loss: 0.2130, Train Accuracy: 0.9140
  Val Loss: 0.2693, Val Accuracy: 0.8979
  New best model saved with accuracy: 0.8979
"""

# %%

# Define inference function for new text
def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()

    # Format the input
    formatted_text = f"Review: {text} Sentiment:"

    # Tokenize
    encoding = tokenizer(formatted_text,
                         return_tensors='pt',
                         max_length=max_length,
                         padding='max_length',
                         truncation=True)

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Get prediction
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        _, preds = torch.max(logits, dim=1)

    # Map prediction to sentiment
    sentiment = "positive" if preds.item() == 1 else "negative"

    return sentiment

# Example usage
sample_texts = [
    "As personal and egoless as you could ever hope to expect from an $120 million self-portrait that doubles as a fable about the fall of Ancient Rome, Francis Ford Coppola’s “Megalopolis” is the story of an ingenious eccentric who dares to stake his fortune on a more optimistic vision for the future — not because he thinks he can single-handedly bring that vision to bear, but rather because history has taught him that questioning a civilization’s present condition is the only reliable hope for preventing its ruin. Needless to say, the movie isn’t arriving a minute too soon.",
    "YALL ARE WRONG You see you gotta go into the movie the same way as Francis Ford Coppola, blazed out of your fucking mind. You gotta meet him on the same plane, the same level of thinking. Hitting the cart 6 times before entering the showing like I did.",
    "the 138-minute cinematic equivalent of someone showing you a youtube video they promise is really good",
    "Dropping the kids off at Oppenheimer so the adults could watch Barbie",
    "Hi Barbie.",
    "As a work of and about plasticity, Barbie succeeds with vibrant, glowing colors. This is a triumph of manufactured design, with its expansive dollhouses and fashion accessories complimenting the mannered, almost manicured narrative journey of toys and humans.",
    "I’m sorry to talk about a man when this is very much a movie about women but every second Gosling is onscreen is so funny. Even when he’s just standing there not talking it’s funny.",
    "A ridiculous achievement in filmmaking. An absurdly immersive and heart-pounding experience. Cillian Murphy is a fucking stud and RDJ will be a front-runner for Best Supporting Actor. Ludwig Göransson put his entire nutsack into that score, coupled with a sound design that made me feel like I took a bomb to the chest."
]
for sample_text in sample_texts:
    prediction = predict_sentiment(sample_text, model, tokenizer, device)
    print(f"Sample text: '{sample_text}'")
    print(f"Predicted sentiment: {prediction}")

"""
Kernel output:

Sample text: 'As personal and egoless as you could ever hope to expect from an $120 million self-portrait that doubles as a fable about the fall of Ancient Rome, Francis Ford Coppola’s “Megalopolis” is the story of an ingenious eccentric who dares to stake his fortune on a more optimistic vision for the future — not because he thinks he can single-handedly bring that vision to bear, but rather because history has taught him that questioning a civilization’s present condition is the only reliable hope for preventing its ruin. Needless to say, the movie isn’t arriving a minute too soon.'
Predicted sentiment: negative
Sample text: 'YALL ARE WRONG You see you gotta go into the movie the same way as Francis Ford Coppola, blazed out of your fucking mind. You gotta meet him on the same plane, the same level of thinking. Hitting the cart 6 times before entering the showing like I did.'
Predicted sentiment: negative
Sample text: 'the 138-minute cinematic equivalent of someone showing you a youtube video they promise is really good'
Predicted sentiment: positive
Sample text: 'Dropping the kids off at Oppenheimer so the adults could watch Barbie'
Predicted sentiment: negative
Sample text: 'Hi Barbie.'
Predicted sentiment: positive
Sample text: 'As a work of and about plasticity, Barbie succeeds with vibrant, glowing colors. This is a triumph of manufactured design, with its expansive dollhouses and fashion accessories complimenting the mannered, almost manicured narrative journey of toys and humans.'
Predicted sentiment: positive
Sample text: 'I’m sorry to talk about a man when this is very much a movie about women but every second Gosling is onscreen is so funny. Even when he’s just standing there not talking it’s funny.'
Predicted sentiment: positive
Sample text: 'A ridiculous achievement in filmmaking. An absurdly immersive and heart-pounding experience. Cillian Murphy is a fucking stud and RDJ will be a front-runner for Best Supporting Actor. Ludwig Göransson put his entire nutsack into that score, coupled with a sound design that made me feel like I took a bomb to the chest.'
Predicted sentiment: positive

Some analysis:

For the reviews which obviously swing in one direction or the other, GPT-2 does
an okay job classifying them. For example, it classifies the last review
accurately. But for less obvious reviews (e.g., the second one, which is clearly
a positive review of Megalopolis), GPT-2 misclassifies. Is this because the
classification model is undertrained, or because GPT-2 itself is not smart
enough? To test this, I pasted in the review in the GPT-2 inference server at
https://huggingface.co/openai-community/gpt2, added in the words "This movie
was", then looked at GPT-2's completion. (I could not just ask GPT-2 if the
review was positive or negative because it's not an instruction-tuned model.) It
said:

YALL ARE WRONG You see you gotta go into the movie the same way as Francis Ford
Coppola, blazed out of your fucking mind. You gotta meet him on the same plane,
the same level of thinking. Hitting the cart 6 times before entering the showing
like I did. This movie was a total fucking disaster. I was so fucking sick of
it. I was so sick of it.

GPT-2 calls the film a "total fucking disaster", so the model really isn't smart
enough to know that review was positive! I asked Claude 3.7 Sonnet, a much
smarter model, if the review was positive or negative (I literally asked it if
was positive or negative), and it said it "appears to be positive, despite its
unconventional style". Good job, Claude!

"""
