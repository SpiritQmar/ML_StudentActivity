import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=10,
                                      n_labels=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

def recommend_activities(person_type, model, activities_list):
    activities = []
    max_recommendations = len(person_type)

    np.random.shuffle(activities_list)

    for i in range(max_recommendations):
        if person_type[i] == 1 and i < len(activities_list):
            activities.append(activities_list[i])

    return activities


activities_list = ["Смотреть сериалы", "Учиться", "Заниматься спортом",
                   "Работать", "Получать повышение", "Развиваться профессионально", "Бухать",
                   "Играть в игры", "Делать уроки", "Участвовать в школьных мероприятиях", "Отдыхать от суеты",
                   "Наслаждаться жизнью"]

new_person = X_test[0].reshape(1, -1)
predicted_type = model.predict(new_person)[0]
recommended_activities = recommend_activities(predicted_type, model, activities_list)
print("Рекомендуемые виды деятельности для Студента сегодня:")
for activity in recommended_activities:
    print("- " + activity)
