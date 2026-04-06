from my_env.tasks import grade_easy, grade_medium, grade_hard

target = 2000

print("Easy:", grade_easy(1900, target))
print("Medium:", grade_medium(1900, target, steps_taken=5, max_steps=10))
print("Hard:", grade_hard(2100, target, steps_taken=8, max_steps=10))