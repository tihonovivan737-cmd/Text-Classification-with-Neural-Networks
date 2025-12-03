import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import re
from collections import Counter
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def tokenize(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()
def preprocess_text(text, vocab, max_len=50):
    tokenized_text = tokenize(text)
    encoded_text = encode_text(tokenized_text, vocab, max_len)
    return torch.tensor(encoded_text, dtype=torch.long).unsqueeze(0).to(device)
def predict_class(text, model, vocab, label_encoder, max_len=50):
    model.eval()
    processed_text = preprocess_text(text, vocab, max_len)
    with torch.no_grad():
        predictions = model(processed_text)
        predicted_class_idx = torch.argmax(predictions, dim=1).item()
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
    return predicted_class
def encode_text(text, vocab, max_len=50):
    encoded = [vocab.get(word, 0) for word in text]
    return encoded[:max_len] + [0] * max(0, max_len - len(encoded))
def check_missing_data(data, label):
    missing_count = np.sum(pd.isnull(data))
    if missing_count > 0:
        print(f"В данных {label} найдено {missing_count} пропущенных значений.")
    else:
        print(f"Пропущенные значения в {label} отсутствуют.")
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, (hidden, _) = self.lstm(embedded)
        final_hidden = hidden[-1]
        normalized_hidden = self.bn(final_hidden)
        return self.fc(normalized_hidden)
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, text):
        embedded = self.embedding(text)
        gru_out, hidden = self.gru(embedded)
        final_hidden = hidden[-1]
        normalized_hidden = self.bn(final_hidden)
        return self.fc(normalized_hidden)
class MLPClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, max_len=50, dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(embedding_dim * max_len, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, text):
        embedded = self.embedding(text)
        flattened = embedded.view(embedded.size(0), -1)
        hidden1 = F.relu(self.bn1(self.fc1(flattened)))
        hidden1 = self.dropout(hidden1)
        hidden2 = F.relu(self.bn2(self.fc2(hidden1)))
        hidden2 = self.dropout(hidden2)
        return self.fc3(hidden2)
class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(maxlen, embed_dim)
    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        return x
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2
class Transformer(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, num_heads, ff_dim, num_classes, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(embed_dim, 20)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(20, num_classes)
    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformer_block(x)
        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = F.softmax(self.dense2(x), dim=-1)
        return x
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    file_path = 'data.txt'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    texts, labels = [], []
    for line in data:
        split_line = line.rsplit(',', 1)
        if len(split_line) == 2:
            text, label = split_line
            texts.append(text.strip())
            labels.append(label.strip())
        else:print(f"Пропуск некорректной строки: {line.strip()}")
    tokenized_texts = [tokenize(text) for text in texts]
    counter = Counter(word for text in tokenized_texts for word in text)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(counter.most_common())}
    vocab['<PAD>'] = 0
    max_len = 50
    encoded_texts = [encode_text(text, vocab, max_len) for text in tokenized_texts]
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(encoded_texts, encoded_labels, test_size=0.2, random_state=42)
    print("Провести анализ данных?")
    print("1: Да")
    print("2: Нет")
    analyze_choice = input("Выбор: ")
    if analyze_choice == "1":
        check_missing_data(X_train, "обучающей выборки (X_train)")
        check_missing_data(X_test, "тестовой выборки (X_test)")
        check_missing_data(y_train, "меток обучающей выборки (y_train)")
        check_missing_data(y_test, "меток тестовой выборки (y_test)")
        print("\nИнформация об обучающей выборке (X_train):")
        print(pd.DataFrame(X_train).info())
        print("\nИнформация о тестовой выборке (X_test):")
        print(pd.DataFrame(X_test).info())
        train_class_counts = Counter(y_train)
        test_class_counts = Counter(y_test)
        print("\nРаспределение классов в обучающей выборке:")
        for label, count in train_class_counts.items():print(f"Класс {label_encoder.inverse_transform([label])[0]}: {count} объектов")
        print("\nРаспределение классов в тестовой выборке:")
        for label, count in test_class_counts.items():print(f"Класс {label_encoder.inverse_transform([label])[0]}: {count} объектов")
        train_labels, train_values = zip(*sorted(train_class_counts.items()))
        test_labels, test_values = zip(*sorted(test_class_counts.items()))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        axes[0].bar(train_labels, train_values, color='blue', alpha=0.7)
        axes[0].set_title("Распределение классов в обучающей выборке")
        axes[0].set_xlabel("Класс")
        axes[0].set_ylabel("Количество объектов")
        axes[0].set_xticks(train_labels)
        axes[0].set_xticklabels(label_encoder.inverse_transform(train_labels), rotation=45)
        axes[1].bar(test_labels, test_values, color='green', alpha=0.7)
        axes[1].set_title("Распределение классов в тестовой выборке")
        axes[1].set_xlabel("Класс")
        axes[1].set_xticks(test_labels)
        axes[1].set_xticklabels(label_encoder.inverse_transform(test_labels), rotation=45)
        plt.tight_layout()
        plt.show()
        text_lengths = [len(text) for text in texts]
    elif analyze_choice == '2':print("Пропуск анализа данных.")
    else:print("Неверный выбор.")
    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_dim = 128
    output_dim = len(label_encoder.classes_)
    pad_idx = vocab['<PAD>']
    print("Выберите модель:")
    print("1: ТРАНСФОРМЕР")
    print("2: MLP")
    print("3: LSTM")
    print("4: GRU")
    analiz_choice = input('Выбор модели: ')
    if analiz_choice == '1':
        model = Transformer(max_len, vocab_size, embedding_dim, num_heads=5, ff_dim=256, num_classes=output_dim).to(device)
    elif analiz_choice == '2':
        model = MLPClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx).to(device)
    elif analiz_choice == '3':
        model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, num_layers=3).to(device)
    elif analiz_choice == '4':
        model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, num_layers=3).to(device)
    else:
        raise ValueError("Неверный выбор модели")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = int(input('Введите количество эпох: '))
    output_folder = 'MLP'
    os.makedirs(output_folder, exist_ok=True)
    output_log_file = os.path.join(output_folder, 'training_log.txt')
    model_save_path = os.path.join(output_folder, 'model.pth')
    datasets_save_path = os.path.join(output_folder, 'datasets.pth')
    vocab_save_path = os.path.join(output_folder, 'vocab.pth')
    label_encoder_save_path = os.path.join(output_folder, 'label_encoder.pth')
    torch.save((train_dataset, test_dataset), datasets_save_path)
    with open(output_log_file, 'w') as f:
        start_time = time.time()
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0
            y_train_true, y_train_pred = [], []
            for texts, labels in train_loader:
                texts, labels = texts.to(device), labels.to(device)
                optimizer.zero_grad()
                predictions = model(texts)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                y_train_true.extend(labels.cpu().numpy())
                y_train_pred.extend(torch.argmax(predictions, axis=1).cpu().numpy())
            train_accuracy = accuracy_score(y_train_true, y_train_pred)
            model.eval()
            y_test_true, y_test_pred = [], []
            with torch.no_grad():
                for texts, labels in test_loader:
                    texts, labels = texts.to(device), labels.to(device)
                    predictions = model(texts)
                    y_test_true.extend(labels.cpu().numpy())
                    y_test_pred.extend(torch.argmax(predictions, axis=1).cpu().numpy())
            test_accuracy = accuracy_score(y_test_true, y_test_pred)
            epoch_loss = train_loss / len(train_loader)
            f.write(f"{epoch + 1},{epoch_loss:.4f},{train_accuracy:.4f},{test_accuracy:.4f}\n")
            print(
                f"Эпоха {epoch + 1}/{n_epochs}, Ошибка: {epoch_loss:.4f}, Точность на обучении: {train_accuracy:.4f}, Точность на тесте: {test_accuracy:.4f}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Обучение завершено. Время затраченное на обучение: {elapsed_time:.2f} секунд")
        torch.save(model.state_dict(), model_save_path)
        torch.save(vocab, vocab_save_path)
        torch.save(label_encoder, label_encoder_save_path)
if __name__ == '__main__':
    main()
