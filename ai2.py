import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from lime.lime_text import LimeTextExplainer

# 讀取資料
data = pd.read_csv('/home/user1/python/filnal_project/Dataset.csv')
data = data.dropna()

# 轉換 sentiment 標籤
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# === 切分資料 ===
train_val_data, test_data = train_test_split(data, test_size=0.1, random_state=42, shuffle=True)
train_data, val_data = train_test_split(train_val_data, test_size=1/9, random_state=42, shuffle=True)
print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

# 建立向量化器與模型
vectorizer = TfidfVectorizer(max_features=5000)
classifier = SGDClassifier(
    loss='log_loss',        # 使用邏輯回歸的 loss
    penalty='l2',           # L2 正則化
    alpha=1e-4,             # 正則化強度
    random_state=42,
    tol=1e-3                
)
model = make_pipeline(vectorizer, classifier)

# 記錄準確率
train_acc_list = []
val_acc_list = []

# 訓練計時
start_time = time.time()

# 執行 5 個 epoch
for epoch in range(1, 6):
    train_sample = train_data.sample(n=min(10000, len(train_data)), random_state=epoch)
    X_train = train_sample['review']
    y_train = train_sample['sentiment']

    val_sample = val_data.sample(n=min(2000, len(val_data)), random_state=epoch)
    X_val = val_sample['review']
    y_val = val_sample['sentiment']

    # 訓練模型
    model.fit(X_train, y_train)

    # 訓練準確率
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    train_acc_list.append(train_acc)

    # 驗證準確率
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch}/5 - Training Accuracy: {train_acc:.4f} | Validation Accuracy: {val_acc:.4f}")

# 結束計時
end_time = time.time()
print(f"\n[Total Training Time]：{end_time - start_time:.2f} seconds")

# 最終測試
X_test = test_data['review']
y_test = test_data['sentiment']
test_preds = model.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)
print(f"\n[Final Test Accuracy]：{test_acc:.4f}")

# 測試新句子的函式
def test_new_sentence(sentence):
    prediction = model.predict([sentence])[0]
    label = "Positive" if prediction == 1 else "Negative"

    print(f"\n[Input Sentence]：{sentence}")
    print(f"[Predicted Sentiment]：{label}")

    # LIME 解釋
    explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
    exp = explainer.explain_instance(sentence, model.predict_proba, num_features=10)

    print("\n[Explanation (重要詞彙貢獻)]：")
    for word, weight in exp.as_list():
        print(f"  {word:20s} => {weight:.4f}")

    print("\n[Full Explanation Text]")
    print(exp.as_text())
