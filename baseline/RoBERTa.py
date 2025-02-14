import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

# 加载本地的RoBERTa模型和分词器
model_name = ""
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = 'cuda:2'  # 选择GPU
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# 定义数据集类
class PostDataset(Dataset):
    def __init__(self, posts, labels, tokenizer, max_len):
        self.posts = posts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.posts)
    
    def __getitem__(self, item):
        # 确保输入文本是字符串类型
        text = str(self.posts[item])  # Convert to string if necessary
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 读取已划分好的数据集
train_df = pd.read_csv('')  # 加载训练集
val_df = pd.read_csv('')  # 加载验证集
test_df = pd.read_csv('')  # 加载测试集

# 标签编码
label_encoder = LabelEncoder()
train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])
val_df['encoded_label'] = label_encoder.transform(val_df['label'])  # 使用相同的编码
test_df['encoded_label'] = label_encoder.transform(test_df['label'])  # 使用相同的编码

# 创建数据集和DataLoader
train_dataset = PostDataset(
    posts=train_df['original_post'].values,
    labels=train_df['encoded_label'].values,
    tokenizer=tokenizer,
    max_len=128
)

val_dataset = PostDataset(
    posts=val_df['original_post'].values,
    labels=val_df['encoded_label'].values,
    tokenizer=tokenizer,
    max_len=128
)

test_dataset = PostDataset(
    posts=test_df['original_post'].values,
    labels=test_df['encoded_label'].values,
    tokenizer=tokenizer,
    max_len=128
)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 冻结模型的其他部分，只训练分类器部分
for param in model.base_model.parameters():
    param.requires_grad = False

# 只训练最后的分类器层
for param in model.classifier.parameters():
    param.requires_grad = True

# 定义训练过程
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 训练过程
for epoch in range(30):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # 验证过程
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            val_preds.extend(preds)
            val_labels.extend(labels)

    val_accuracy = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch + 1} - Validation Accuracy: {val_accuracy}")

# 测试过程
model.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        test_preds.extend(preds)
        test_labels.extend(labels)

# 计算总体指标
test_accuracy = accuracy_score(test_labels, test_preds)
test_macro_f1 = f1_score(test_labels, test_preds, average='macro')
test_micro_f1 = f1_score(test_labels, test_preds, average='micro')

# 打印总体指标
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print(f"Test Micro F1: {test_micro_f1:.4f}")

# 计算标签0和1的precision, recall, F1
# 不再使用pos_label参数，改为使用labels参数
precision_0 = precision_score(test_labels, test_preds, labels=[0], average='binary')
recall_0 = recall_score(test_labels, test_preds, labels=[0], average='binary')
f1_0 = f1_score(test_labels, test_preds, labels=[0], average='binary')

precision_1 = precision_score(test_labels, test_preds, labels=[1], average='binary')
recall_1 = recall_score(test_labels, test_preds, labels=[1], average='binary')
f1_1 = f1_score(test_labels, test_preds, labels=[1], average='binary')

print(f"\nMetrics for class 0:")
print(f"Precision (0): {precision_0:.4f}")
print(f"Recall (0): {recall_0:.4f}")
print(f"F1 (0): {f1_0:.4f}")

print(f"\nMetrics for class 1:")
print(f"Precision (1): {precision_1:.4f}")
print(f"Recall (1): {recall_1:.4f}")
print(f"F1 (1): {f1_1:.4f}")

# 按 field 计算指标
field_groups = test_df.groupby('field')
for field, group in field_groups:
    field_indices = test_df[test_df['field'] == field].index
    field_preds = [test_preds[i] for i in field_indices]
    field_labels = [test_labels[i] for i in field_indices]
    
    # 计算准确率
    field_accuracy = accuracy_score(field_labels, field_preds)
    
    # 手动计算TP, FP, FN
    TP = sum(1 for p, l in zip(field_preds, field_labels) if p == 1 and l == 1)
    FP = sum(1 for p, l in zip(field_preds, field_labels) if p == 1 and l == 0)
    FN = sum(1 for p, l in zip(field_preds, field_labels) if p == 0 and l == 1)
    
    # 手动计算precision和recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # 手动计算F1
    field_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nField: {field}")
    print(f"Field Accuracy: {field_accuracy:.4f}")
    print(f"Field F1: {field_f1:.4f}")

# 按 source 计算指标
source_groups = test_df.groupby('source')
for source, group in source_groups:
    source_indices = test_df[test_df['source'] == source].index
    source_preds = [test_preds[i] for i in source_indices]
    source_labels = [test_labels[i] for i in source_indices]

    source_accuracy = accuracy_score(source_labels, source_preds)
    source_macro_f1 = f1_score(source_labels, source_preds, average='macro')

    print(f"\nSource: {source}")
    print(f"Source Accuracy: {source_accuracy:.4f}")
    print(f"Source Macro F1: {source_macro_f1:.4f}")
