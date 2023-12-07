import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 데이터 로드
filename = "/Users/noooo/Desktop/Aviation safety.csv"
column_names = ['airline', 'avail_seat_km_per_week', 'incidents_85_14', 'fatal_accident_85_14', 'fatalities_85_14', 'target']
data = pd.read_csv(filename, names=column_names)

# 데이터 정보 확인
print(data.info())

# 결측치 확인
print("Missing Values:\n", data.isnull().sum())

# 기술 통계량 확인
print("Descriptive Statistics:\n", data.describe())

# 항공사별 사고발생 현황 시각화
plt.figure(figsize=(12, 6))
sns.countplot(x='airline', data=data, hue='target')
plt.title('Accident Occurrence by Airline')
plt.show()

# 연도별 사고 및 사망자 수 추이 시각화
data['year'] = data['incidents_85_14'].astype(str).str[:2]
plt.figure(figsize=(12, 6))
sns.countplot(x='year', data=data, hue='target')
plt.title('Accident and Fatality Trends by Year')
plt.show()

# 변수 간의 상관 관계 분석을 위한 히트맵 작성
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# 데이터 전처리
# 레이블 인코딩
label_encoder = LabelEncoder()
data['airline'] = label_encoder.fit_transform(data['airline'])

# 변수 간 스케일링
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop(['target', 'year'], axis=1))
scaled_data = pd.DataFrame(scaled_data, columns=data.columns[:-2])

# 머신러닝 모델 선택
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(scaled_data, data['target'], test_size=0.2, random_state=42)

# 모델 학습
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 성능 검증
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(model, scaled_data, data['target'], cv=kfold)
print("Cross-validated Accuracy:", cv_results.mean())