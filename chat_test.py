from app.chat import ask_question

print("ML Debugger Chat (type 'exit' to stop)\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    answer = ask_question(question)
    print("\nAI:", answer, "\n")