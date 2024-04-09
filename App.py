import numpy as np
from sklearn.tree import DecisionTreeClassifier

from main import X_train, y_train, activities_list




def recommend_activities_for_day(person_type, model, activities_list, day):
    activities = []
    max_recommendations = len(person_type)

    np.random.shuffle(activities_list)

    for i in range(max_recommendations):
        if person_type[i] == 1 and i < len(activities_list):
            activities.append(f"{activities_list[i]} ({day})")  

    return activities

def main():

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)


    new_person = np.random.rand(1, 10)  


    predicted_type = model.predict(new_person)[0]


    days = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
    for day in days:
        recommended_activities = recommend_activities_for_day(predicted_type, model, activities_list, day)
        print(f"Рекомендуемые виды деятельности на {day}:")
        for activity in recommended_activities:
            print("- " + activity)

if __name__ == "__main__":
    main()
