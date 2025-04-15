import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 讀取試算表資料
df = pd.read_csv("Taipei_house.csv")

# 簡單清理（根據你檔案內容再做詳細處理）
df.dropna(inplace=True)

# 特徵選擇
features = ['region', 'building_age', 'building_type', 'floor', 'near_mrt', 'total_ping']
X = df[features]
y = df['target_price']  # 預測目標

# One-hot encoding 文字欄位
X = pd.get_dummies(X)

# 數據標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分訓練/測試資料
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 建立模型
model = XGBRegressor()
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"預測RMSE: {rmse:.2f} 萬元")
