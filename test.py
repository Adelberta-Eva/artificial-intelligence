import pandas as pd
from sklearn.model_selection import train_test_split
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification
from torch.nn import init
from transformers import Trainer, TrainingArguments

# 禁用 oneDNN 操作，减少数值不一致问题
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 加载英文数据集
df_en = pd.read_csv("C:/Users/HONOR/Desktop/imdb_top_1000.csv")
df_en = df_en[['Overview', 'IMDB_Rating']]
df_en = df_en.dropna()

# 简单的标签化函数（英文数据集）
def label_sentiment(rating):
    if rating > 7:
        return 2  # Positive
    elif rating >= 5:
        return 1  # Neutral
    else:
        return 0  # Negative

df_en['Sentiment'] = df_en['IMDB_Rating'].apply(label_sentiment)

# 拆分英文数据集
train_texts_en, test_texts_en, train_labels_en, test_labels_en = train_test_split(df_en['Overview'], df_en['Sentiment'], test_size=0.2, random_state=42)

# 加载BERT模型与tokenizer
bert_model_path = "bert-base-uncased"
tokenizer_en = BertTokenizer.from_pretrained(bert_model_path)
model_en = BertForSequenceClassification.from_pretrained(bert_model_path, num_labels=3)

# 加载中文数据集
df_cn = pd.read_csv("C:/Users/HONOR/Desktop/douban_movies.csv", encoding='gbk')
df_cn = df_cn[['Movie Name', 'Intro', 'Rating']]
df_cn = df_cn.dropna()

# 简单的标签化函数（中文数据集）
df_cn['Sentiment'] = df_cn['Rating'].apply(label_sentiment)

# 拆分中文数据集
train_texts_cn, test_texts_cn, train_labels_cn, test_labels_cn = train_test_split(df_cn['Intro'], df_cn['Sentiment'], test_size=0.2, random_state=42)

# 加载GPT-2模型与tokenizer
from transformers import GPT2Tokenizer, GPT2Model
# 加载GPT-2模型与tokenizer
# 加载 GPT-2 模型与 tokenizer
gpt2_model_path = "gpt2"
tokenizer_cn = GPT2Tokenizer.from_pretrained(gpt2_model_path)


# 设置 pad_token 为 eos_token
tokenizer_cn.pad_token = tokenizer_cn.eos_token  # 使用 eos_token 作为 pad_token

model_cn = GPT2ForSequenceClassification.from_pretrained(gpt2_model_path, num_labels=3)
model_cn.config.pad_token_id = tokenizer_cn.pad_token_id  # 明确设置 pad_token_id

# 重新初始化分类器层权重
def reinitialize_classifier(model):
    if isinstance(model, BertForSequenceClassification):
        classifier = model.classifier
        if classifier is not None:
            init.xavier_uniform_(classifier.weight)
            init.zeros_(classifier.bias)
    elif isinstance(model, GPT2ForSequenceClassification):
        # 确保 GPT2 模型没有 'score' 层
        if hasattr(model, 'score') and model.score is not None:
            print("Warning: GPT-2 model already has a 'score' layer.")
        else:
            # 如果没有 'score' 层，我们添加它
            model.score = torch.nn.Linear(model.config.n_embd, model.config.num_labels)
            init.xavier_uniform_(model.score.weight)
            init.zeros_(model.score.bias)

# 初始化BERT和GPT-2的分类器
reinitialize_classifier(model_en)
reinitialize_classifier(model_cn)

# 确保 GPT-2 的 forward 函数能正确处理批量数据
# 编码时确保 padding 参数为 True，特别是在批量处理时
def encode_texts(texts, tokenizer, max_length=512):
    return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors='pt')


# 编码英文和中文文本
train_encodings_en = encode_texts(train_texts_en, tokenizer_en)
test_encodings_en = encode_texts(test_texts_en, tokenizer_en)
train_encodings_cn = encode_texts(train_texts_cn, tokenizer_cn)
test_encodings_cn = encode_texts(test_texts_cn, tokenizer_cn)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # 确保每个输入和标签的对齐
        item = {key: torch.tensor(val[idx]).clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx]).clone().detach()  # 使用 .clone().detach() 来避免不必要的计算图
        return item

    def __len__(self):
        return len(self.labels)


# 创建英文和中文数据集
train_dataset_en = SentimentDataset(train_encodings_en, train_labels_en)
test_dataset_en = SentimentDataset(test_encodings_en, test_labels_en)
train_dataset_cn = SentimentDataset(train_encodings_cn, train_labels_cn)
test_dataset_cn = SentimentDataset(test_encodings_cn, test_labels_cn)

# 根据数据集大小动态设置批量大小
batch_size = 8 if len(train_texts_cn) < 500 else 16

print(len(train_encodings_en['input_ids']), len(train_labels_en))  # 检查英文数据集的长度
print(len(train_encodings_cn['input_ids']), len(train_labels_cn))  # 检查中文数据集的长度


# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=3,              # 训练周期
    per_device_train_batch_size=batch_size,  # 每个设备的批次大小
    per_device_eval_batch_size=64,   # 每个设备的评估批次大小
    warmup_steps=500,                # 学习率预热步数
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    logging_steps=10,
    gradient_accumulation_steps=2,   # 梯度累积步数
)

# 训练英文模型
trainer_en = Trainer(
    model=model_en,
    args=training_args,
    train_dataset=train_dataset_en,
    eval_dataset=test_dataset_en,
    tokenizer=tokenizer_cn  # 传递 tokenizer 确保 padding 正确
)

trainer_en.train()

# 训练中文模型
trainer_cn = Trainer(
    model=model_cn,
    args=training_args,
    train_dataset=train_dataset_cn,
    eval_dataset=test_dataset_cn
)

trainer_cn.train()

# 预测函数
sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

def predict_sentiment(text, model, tokenizer):
    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    output = model(**encoding)
    logits = output.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return sentiment_labels[prediction]

# 示例：对英文电影简介进行情感预测
movie_overview_en = "A thrilling adventure of a group of explorers discovering new lands."
sentiment_en = predict_sentiment(movie_overview_en, model_en, tokenizer_en)
print(f"English Sentiment: {sentiment_en}")

# 示例：对中文电影简介进行情感预测
movie_overview_cn = "一群探险家发现了新的土地。"
sentiment_cn = predict_sentiment(movie_overview_cn, model_cn, tokenizer_cn)
print(f"Chinese Sentiment: {sentiment_cn}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#模型评估
# 预测函数 - 批量预测
def predict_on_dataset(model, tokenizer, dataset):
    model.eval()  # 设置模型为评估模式
    predictions = []
    true_labels = []

    for batch in torch.utils.data.DataLoader(dataset, batch_size=64):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            preds = torch.argmax(logits, dim=-1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels


# 获取英文模型的预测结果
predictions_en, true_labels_en = predict_on_dataset(model_en, tokenizer_en, test_dataset_en)

# 获取中文模型的预测结果
predictions_cn, true_labels_cn = predict_on_dataset(model_cn, tokenizer_cn, test_dataset_cn)

# 计算英文模型的评估指标
accuracy_en = accuracy_score(true_labels_en, predictions_en)
precision_en = precision_score(true_labels_en, predictions_en, average='weighted')
recall_en = recall_score(true_labels_en, predictions_en, average='weighted')
f1_en = f1_score(true_labels_en, predictions_en, average='weighted')

# 计算中文模型的评估指标
accuracy_cn = accuracy_score(true_labels_cn, predictions_cn)
precision_cn = precision_score(true_labels_cn, predictions_cn, average='weighted')
recall_cn = recall_score(true_labels_cn, predictions_cn, average='weighted')
f1_cn = f1_score(true_labels_cn, predictions_cn, average='weighted')

# 输出英文模型评估结果
print("English Model Evaluation Results:")
print(f"Accuracy: {accuracy_en:.4f}")
print(f"Precision: {precision_en:.4f}")
print(f"Recall: {recall_en:.4f}")
print(f"F1-Score: {f1_en:.4f}")

# 输出中文模型评估结果
print("\nChinese Model Evaluation Results:")
print(f"Accuracy: {accuracy_cn:.4f}")
print(f"Precision: {precision_cn:.4f}")
print(f"Recall: {recall_cn:.4f}")
print(f"F1-Score: {f1_cn:.4f}")
