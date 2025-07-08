import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split # разделение данных на train/test
from sklearn.preprocessing import StandardScaler #Нормализация данных

iris = load_iris()  # Загружаем датасет
x = iris.data       # Данные 150 на 4
y = iris.target     # Метки 0, 1, 2

# нормируем даныне по формуле X_new = (X - mean) / std
# где mean - среднее значение
# std стандартное отклонение
scaler = StandardScaler()
x = scaler.fit_transform(x)

#разделение данных на train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#конвертирую в тензоры PyTorch
x_train = torch.FloatTensor(x_train)  # Данные → float32
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)   # Метки → int64
y_test = torch.LongTensor(y_test)

#архитектура модели
class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.layer1 = nn.Linear(4, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
      x = self.relu(self.layer1(x)) # Проход через 1-й слой + активация
      x = self.relu(self.layer2(x))
      x = self.output(x) # Выходной слой (без активации!)
      return x

model = IrisModel()
criterion = nn.CrossEntropyLoss()  # Для классификации
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Оптимизатор

for epoch in range(200):
  #Модель делает предсказания для X_train.
  #Сравнивает их с y_train через CrossEntropyLoss.
  outputs = model(x_train)
  loss = criterion(outputs, y_train)

  optimizer.zero_grad()  # Обнуляем градиенты
  loss.backward()        # Вычисляем градиенты (как нужно изменить веса, чтобы уменьшить ошибку).
  optimizer.step()       # Обновляем веса


with torch.no_grad():  # Отключаем вычисление градиентов (только предсказание)
    predictions = model(x_test)
    _, predicted = torch.max(predictions, 1)  # Берём индекс класса с максимальной вероятностью
    accuracy = (predicted == y_test).sum().item() / y_test.size(0) #процент правильных ответов на тестовых данных.
    print(f'Accuracy: {accuracy * 100:.2f}%')
# тест предсказание для нового цветка
new_flower = torch.FloatTensor([[5.1, 3.5, 1.4, 0.2]])  # Данные нового цветка
new_flower = scaler.transform(new_flower) # Нормализуем как обучающие данные
new_flower = torch.FloatTensor(new_flower)

with torch.no_grad():
    prediction = model(new_flower)
    predicted_class = torch.argmax(prediction).item()
    print(f"Это {iris.target_names[predicted_class]}")
